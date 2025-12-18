# agent_expectimax.py
import math
import random
import logic
import constants as c

MOVE_FUNCS = {
    "Up": logic.up,
    "Down": logic.down,
    "Left": logic.left,
    "Right": logic.right,
}


def _clone(mat):
    return [row[:] for row in mat]


def _empty_cells(mat):
    res = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                res.append((i, j))
    return res


def _apply_special_cell_effect(mat, special_pos):
    """改版规则：成功移动后，特殊格子上的值如果 >2，则整除2。"""
    if special_pos is None:
        return mat
    i, j = special_pos
    if mat[i][j] > 2:
        mat[i][j] //= 2
    return mat


def _add_two_at(mat, pos):
    """你的版本：新 tile 只会生成 2。"""
    i, j = pos
    mat[i][j] = 2
    return mat


def _log2(v):
    return 0 if v <= 0 else int(math.log2(v))


# ==================== 可调参数区域 ====================
# 启发式函数权重参数（影响评估准确性）
HEURISTIC_WEIGHTS = {
    'empty': 50,          # 空格权重（最重要，建议范围：1000-1500）
    'max_tile': 2000,        # 最大块权重（建议范围：200-300）
    'smooth': 8,            # 平滑度权重（建议范围：5-15）
    'mono_h': 3,            # 行单调性权重（建议范围：2-5）
    'mono_v': 3,            # 列单调性权重（建议范围：2-5）
    'corner_main': 10,      # 主角落（左上）权重（建议范围：8-15）
    'corner_other': 3,       # 其他角落权重（建议范围：2-5）
    'merge_potential': 5,   # 合并潜力权重（建议范围：10-20）
    'special_penalty': 100,    # 特殊格子惩罚系数（建议范围：3-8）
    'special_penalty_scale': 2.0,  # 特殊格子惩罚缩放（大数字惩罚更重，建议范围：1.5-3.0）
    'special_threshold': 32,  # 特殊格子严重惩罚阈值（>=此值惩罚更重，建议范围：16-64）
    'merge_special_reduction': 0.5,  # 特殊格子合并奖励减少比例（0.0-1.0，建议范围：0.2-0.5）
    'merge_special_threshold': 16,  # 特殊格子合并减少阈值（>=此值才减少，建议范围：8-32）
    'merge_special_penalty': 50,  # 特殊格子合并额外惩罚（建议范围：20-100）
    'empty_critical_bonus': 2000,  # 空格数<=3时的额外奖励（建议范围：1000-5000）
    'empty_critical_threshold': 3,  # 空格数临界阈值（<=此值才给予额外奖励）
    'max_tile_special_penalty': 5000000,  # 最大块在特殊格子的严重惩罚（建议范围：3000-10000）
    'max_tile_protection_threshold': 16,  # 最大块保护阈值（>=此值才保护，建议范围：8-32）
    'half_max_tile_special_penalty': 30000,  # 最大块/2在特殊格子的严重惩罚（建议范围：2000-6000）
    'half_max_tile_protection_threshold': 8,  # 最大块/2保护阈值（>=此值才保护，建议范围：4-16）
}

# 危险位置评估参数
DANGER_PARAMS = {
    'corner_danger': 3,      # 角落危险分数（建议范围：2-5）
    'edge_danger': 1,        # 边缘危险分数（建议范围：1-2）
    'big_neighbor_threshold': 8,  # 大数字邻居阈值（建议范围：4-16）
    'big_neighbor_penalty': 2,   # 大数字邻居惩罚（建议范围：1-3）
}
# =====================================================


def _heuristic(mat, special_pos=None):
    """
    优化的启发式评估函数：越大越好（站在玩家角度）。
    包含：空格、最大块、平滑度、单调性、角落策略、合并潜力、特殊格子考虑。
    """
    n = len(mat)
    logs = [[_log2(mat[i][j]) for j in range(n)] for i in range(n)]

    # 1. 空格数量（最重要）
    empty = sum(1 for i in range(n) for j in range(n) if mat[i][j] == 0)
    
    # 1.1 空格数临界奖励：当空格数 <= 3 时，给予额外奖励（鼓励保持空格）
    empty_critical_bonus = 0
    if empty <= HEURISTIC_WEIGHTS['empty_critical_threshold']:
        # 空格数越少，奖励越大（鼓励保持空格）
        empty_critical_bonus = HEURISTIC_WEIGHTS['empty_critical_bonus'] * (HEURISTIC_WEIGHTS['empty_critical_threshold'] + 1 - empty)
    
    # 2. 最大块
    max_tile = max(max(row) for row in mat)
    max_log = _log2(max_tile)
    
    # 2.1 最大块位置保护：找到最大块和最大块/2的位置，如果它们在特殊格子上，给予严重惩罚
    # 记录场上最大tile的值，用于动态保护
    max_tile_pos = None
    half_max_tile_pos = None
    half_max_tile = max_tile // 2  # 最大块的一半（如果最大块是64，则half_max_tile是32）
    
    for i in range(n):
        for j in range(n):
            if mat[i][j] == max_tile:
                max_tile_pos = (i, j)
            # 如果最大块/2存在且大于0，也记录它的位置
            if mat[i][j] == half_max_tile and half_max_tile > 0:
                half_max_tile_pos = (i, j)
    
    # 3. 平滑度：相邻格子差值越小越好
    smooth = 0
    for i in range(n):
        for j in range(n):
            if mat[i][j] != 0:
                if i + 1 < n and mat[i + 1][j] != 0:
                    smooth -= abs(logs[i][j] - logs[i + 1][j])
                if j + 1 < n and mat[i][j + 1] != 0:
                    smooth -= abs(logs[i][j] - logs[i][j + 1])
    
    # 4. 单调性：鼓励行列单调递增/递减（更容易合并）
    # 行单调性（从左到右或从右到左）
    mono_h = 0
    for i in range(n):
        inc = 0
        dec = 0
        for j in range(n - 1):
            if mat[i][j] != 0 and mat[i][j + 1] != 0:
                if logs[i][j] >= logs[i][j + 1]:
                    inc += logs[i][j] - logs[i][j + 1]
                else:
                    dec += logs[i][j + 1] - logs[i][j]
        mono_h += max(inc, dec)
    
    # 列单调性（从上到下或从下到上）
    mono_v = 0
    for j in range(n):
        inc = 0
        dec = 0
        for i in range(n - 1):
            if mat[i][j] != 0 and mat[i + 1][j] != 0:
                if logs[i][j] >= logs[i + 1][j]:
                    inc += logs[i][j] - logs[i + 1][j]
                else:
                    dec += logs[i + 1][j] - logs[i][j]
        mono_v += max(inc, dec)
    
    # 5. 角落策略：鼓励大数字在角落（左上角优先）
    corner_bonus = 0
    if mat[0][0] != 0:
        corner_bonus += logs[0][0] * HEURISTIC_WEIGHTS['corner_main']
    # 也考虑其他角落，但权重较低
    if mat[0][n-1] != 0:
        corner_bonus += logs[0][n-1] * HEURISTIC_WEIGHTS['corner_other']
    if mat[n-1][0] != 0:
        corner_bonus += logs[n-1][0] * HEURISTIC_WEIGHTS['corner_other']
    if mat[n-1][n-1] != 0:
        corner_bonus += logs[n-1][n-1] * HEURISTIC_WEIGHTS['corner_other']
    
    # 6. 合并潜力：评估可能的合并机会
    merge_potential = 0
    for i in range(n):
        for j in range(n):
            if mat[i][j] != 0:
                # 检查上下左右是否有相同值的格子
                # 向下合并：合并后结果在 (i, j)，(i+1, j) 变为0
                if i + 1 < n and mat[i + 1][j] == mat[i][j]:
                    merge_reward = logs[i][j] * 2
                    # 如果涉及特殊格子且数字较大，减小合并奖励
                    if special_pos is not None:
                        # 检查：合并的两个格子是否在特殊格子上，或合并后结果在特殊格子上
                        is_special_involved = (
                            (i, j) == special_pos or      # 合并前位置1在特殊格子
                            (i + 1, j) == special_pos     # 合并前位置2在特殊格子
                            # 合并后结果在 (i, j)，如果 (i, j) 是特殊格子，上面已检查
                        )
                        if is_special_involved and mat[i][j] >= HEURISTIC_WEIGHTS['merge_special_threshold']:
                            # 减小合并奖励（大数字经过特殊格子后合并价值降低，因为会被减半）
                            reduction = HEURISTIC_WEIGHTS['merge_special_reduction']
                            merge_reward *= (1.0 - reduction)
                            # 额外惩罚：进入特殊格后合成的额外惩罚
                            merge_reward -= HEURISTIC_WEIGHTS['merge_special_penalty']
                    merge_potential += merge_reward
                
                # 向右合并：合并后结果在 (i, j)，(i, j+1) 变为0
                if j + 1 < n and mat[i][j + 1] == mat[i][j]:
                    merge_reward = logs[i][j] * 2
                    # 如果涉及特殊格子且数字较大，减小合并奖励
                    if special_pos is not None:
                        is_special_involved = (
                            (i, j) == special_pos or      # 合并前位置1在特殊格子
                            (i, j + 1) == special_pos    # 合并前位置2在特殊格子
                        )
                        if is_special_involved and mat[i][j] >= HEURISTIC_WEIGHTS['merge_special_threshold']:
                            # 减小合并奖励（大数字经过特殊格子后合并价值降低）
                            reduction = HEURISTIC_WEIGHTS['merge_special_reduction']
                            merge_reward *= (1.0 - reduction)
                            # 额外惩罚：进入特殊格后合成的额外惩罚
                            merge_reward -= HEURISTIC_WEIGHTS['merge_special_penalty']
                    merge_potential += merge_reward
    
    # 7. 特殊格子惩罚：如果特殊格子上有大数字，给予惩罚（因为可能被减半）
    special_penalty = 0
    max_tile_special_penalty = 0  # 最大块在特殊格子的额外严重惩罚
    half_max_tile_special_penalty = 0  # 最大块/2在特殊格子的额外严重惩罚
    
    if special_pos is not None:
        i, j = special_pos
        special_tile_value = mat[i][j]
        
        if special_tile_value > 2:
            # 基础惩罚
            base_penalty = logs[i][j] * HEURISTIC_WEIGHTS['special_penalty']
            # 如果数字很大（>=阈值），使用缩放惩罚（更严重）
            if special_tile_value >= HEURISTIC_WEIGHTS['special_threshold']:
                # 对大数字使用更重的惩罚（平方或缩放）
                penalty_multiplier = HEURISTIC_WEIGHTS['special_penalty_scale']
                special_penalty = -base_penalty * penalty_multiplier
            else:
                special_penalty = -base_penalty
            
            # 7.1 最大块保护：如果最大块在特殊格子上，给予非常严重的惩罚
            # 动态检查：如果特殊格子上的值等于当前场上最大tile
            if special_tile_value == max_tile:
                if max_tile >= HEURISTIC_WEIGHTS['max_tile_protection_threshold']:
                    # 最大块在特殊格子上，给予严重惩罚（防止最大块掉入特殊格子）
                    max_tile_special_penalty = -HEURISTIC_WEIGHTS['max_tile_special_penalty']
            
            # 7.2 最大块/2保护：如果最大块/2在特殊格子上，也给予严重惩罚
            # 动态检查：如果特殊格子上的值等于当前场上最大tile的一半
            # 例如：如果最大tile是64，则32落入特殊格也会受到严重惩罚
            if special_tile_value == half_max_tile and half_max_tile > 0:
                if half_max_tile >= HEURISTIC_WEIGHTS['half_max_tile_protection_threshold']:
                    # 最大块/2在特殊格子上，给予严重惩罚（防止最大块/2掉入特殊格子）
                    half_max_tile_special_penalty = -HEURISTIC_WEIGHTS['half_max_tile_special_penalty']
    
    # 8. 加权求和（使用可调权重）
    return (
        empty * HEURISTIC_WEIGHTS['empty']
        + empty_critical_bonus  # 空格数临界奖励
        + max_log * HEURISTIC_WEIGHTS['max_tile']
        + smooth * HEURISTIC_WEIGHTS['smooth']
        + mono_h * HEURISTIC_WEIGHTS['mono_h'] + mono_v * HEURISTIC_WEIGHTS['mono_v']
        + corner_bonus
        + merge_potential * HEURISTIC_WEIGHTS['merge_potential']
        + special_penalty
        + max_tile_special_penalty  # 最大块保护惩罚
        + half_max_tile_special_penalty  # 最大块/2保护惩罚
    )


class ExpectimaxAgent:
    """
    优化的 Expectimax Agent：
    - Max层：玩家选方向（最大化期望得分）
    - Chance层：随机生成新 tile 的位置（期望值计算）
    - 改进的启发式函数和更智能的 chance 节点选择
    """

    def __init__(self, depth=2, special_pos=None, max_chance_branches=6, sample_chance=False):
        """
        可调参数说明：
        depth: 搜索深度（建议范围：2-5）
            - 增加：更准确但更慢（指数级增长）
            - 减少：更快但可能不够准确
            - 推荐：3-4（平衡点）
        
        max_chance_branches: chance 节点分支数限制（建议范围：4-10）
            - 增加：更准确但更慢
            - 减少：更快但可能忽略重要情况
            - 推荐：6-8
        
        sample_chance: 是否使用采样模式
            - True: 随机采样，更快但有随机性
            - False: 智能选择危险位置，更保守但更稳定
            - 推荐：False（更稳定）
        """
        self.depth = depth
        self.special_pos = special_pos
        self.max_chance_branches = max_chance_branches
        self.sample_chance = sample_chance

    def _get_available_moves(self, mat):
        """
        获取所有可用的移动方向。
        返回：[(move_name, move_fn, next_mat, done), ...]
        """
        available = []
        for move_name, move_fn in MOVE_FUNCS.items():
            next_mat, done = move_fn(_clone(mat))
            if done:
                # 应用特殊格子减半
                next_mat = _apply_special_cell_effect(next_mat, self.special_pos)
                available.append((move_name, move_fn, next_mat, True))
        return available
    
    def _is_board_full(self, mat):
        """检查棋盘是否已填满（没有空格）。"""
        n = len(mat)
        for i in range(n):
            for j in range(n):
                if mat[i][j] == 0:
                    return False
        return True
    
    def _count_merges_in_move(self, mat, move_fn):
        """
        计算某个移动方向可以产生的合并数量。
        用于在棋盘填满时识别可以合并的方向。
        方法：比较移动前后的矩阵，统计合并产生的空格数量。
        """
        next_mat, done = move_fn(_clone(mat))
        if not done:
            return 0
        
        # 计算移动后产生的空格数量（合并会产生空格）
        n = len(mat)
        original_empty = sum(1 for i in range(n) for j in range(n) if mat[i][j] == 0)
        next_empty = sum(1 for i in range(n) for j in range(n) if next_mat[i][j] == 0)
        
        # 空格增加的数量 = 合并的数量（每次合并产生1个空格）
        merge_count = next_empty - original_empty
        
        # 如果空格没有增加，但矩阵有变化，说明只是移动没有合并
        # 但这种情况在填满的棋盘上不应该发生（因为填满时移动必须产生合并）
        return max(0, merge_count)
    
    def _has_merge_potential(self, mat, move_fn):
        """
        检查某个移动方向是否可能产生合并。
        更简单的方法：检查移动后是否有空格产生。
        """
        next_mat, done = move_fn(_clone(mat))
        if not done:
            return False
        
        # 如果移动后产生了空格，说明有合并
        n = len(mat)
        original_empty = sum(1 for i in range(n) for j in range(n) if mat[i][j] == 0)
        next_empty = sum(1 for i in range(n) for j in range(n) if next_mat[i][j] == 0)
        
        return next_empty > original_empty
    
    def choose_move(self, mat):
        """
        返回 'Up'/'Down'/'Left'/'Right' 或 None（无路可走）。
        优化：当棋盘填满时，优先考虑可以产生合并的方向。
        """
        # 检查棋盘是否填满
        is_full = self._is_board_full(mat)
        
        # 获取所有可用的移动
        available_moves = self._get_available_moves(mat)
        
        if not available_moves:
            return None
        
        # 如果棋盘填满，优先考虑可以产生合并的方向
        if is_full:
            # 筛选出可以产生合并的方向
            moves_with_merges = []
            for move_name, move_fn, next_mat, done in available_moves:
                if self._has_merge_potential(mat, move_fn):
                    merge_count = self._count_merges_in_move(mat, move_fn)
                    moves_with_merges.append((merge_count, move_name, next_mat))
            
            if moves_with_merges:
                # 按合并数量排序（合并数量多的优先）
                moves_with_merges.sort(reverse=True, key=lambda x: x[0])
                
                # 在可以合并的方向中选择最优的
                best_move = None
                best_score = -float("inf")
                
                for merge_count, move_name, next_mat in moves_with_merges:
                    score = self._chance_node(next_mat, self.depth - 1)
                    if score > best_score:
                        best_score = score
                        best_move = move_name
                
                if best_move is not None:
                    return best_move
            # 如果没有可以合并的方向，继续正常流程（虽然理论上不应该发生）
        
        # 正常情况：在所有可用方向中选择最优的
        best_move = None
        best_score = -float("inf")
        
        for move_name, move_fn, next_mat, done in available_moves:
            score = self._chance_node(next_mat, self.depth - 1)
            if score > best_score:
                best_score = score
                best_move = move_name
        
        return best_move

    def _max_node(self, mat, depth):
        """Max层：玩家选方向。"""
        state = logic.game_state(mat)
        if depth <= 0 or state != "not over":
            return _heuristic(mat, self.special_pos)

        best = -float("inf")
        moved = False

        for move_name, move_fn in MOVE_FUNCS.items():
            next_mat, done = move_fn(_clone(mat))
            if not done:
                continue
            moved = True
            next_mat = _apply_special_cell_effect(next_mat, self.special_pos)
            best = max(best, self._chance_node(next_mat, depth - 1))

        if not moved:
            return _heuristic(mat, self.special_pos)
        return best

    def _get_dangerous_positions(self, mat, empties):
        """
        识别"危险"位置：角落、边缘、或会阻碍合并的位置。
        这些位置如果出现新 tile，通常更不利。
        """
        n = len(mat)
        dangerous = []
        
        for i, j in empties:
            danger_score = 0
            
            # 角落位置通常更危险（除非是目标角落左上角）
            if (i, j) == (0, 0):
                # 左上角是目标位置，不算危险
                pass
            elif (i == 0 and j == n-1) or (i == n-1 and j == 0) or (i == n-1 and j == n-1):
                # 其他三个角落更危险
                danger_score += DANGER_PARAMS['corner_danger']
            
            # 边缘位置（非角落）
            if (i == 0 or i == n-1 or j == 0 or j == n-1) and (i, j) != (0, 0):
                danger_score += DANGER_PARAMS['edge_danger']
            
            # 如果周围有大数字，更危险
            neighbors = []
            if i > 0 and mat[i-1][j] != 0:
                neighbors.append(mat[i-1][j])
            if i < n-1 and mat[i+1][j] != 0:
                neighbors.append(mat[i+1][j])
            if j > 0 and mat[i][j-1] != 0:
                neighbors.append(mat[i][j-1])
            if j < n-1 and mat[i][j+1] != 0:
                neighbors.append(mat[i][j+1])
            
            if neighbors:
                max_neighbor = max(neighbors)
                if max_neighbor >= DANGER_PARAMS['big_neighbor_threshold']:
                    danger_score += DANGER_PARAMS['big_neighbor_penalty']
            
            dangerous.append((danger_score, (i, j)))
        
        # 按危险程度排序（越危险越靠前）
        dangerous.sort(reverse=True, key=lambda x: x[0])
        return [pos for _, pos in dangerous]

    def _chance_node(self, mat, depth):
        """
        优化的 Chance 节点：更智能地选择候选位置。
        优先考虑更危险的位置（角落、边缘），因为这些位置出现新 tile 通常更不利。
        """
        state = logic.game_state(mat)
        if depth <= 0 or state != "not over":
            return _heuristic(mat, self.special_pos)

        empties = _empty_cells(mat)
        if not empties:
            return _heuristic(mat, self.special_pos)

        # 如果空位不多，全部考虑（精确计算期望值）
        if len(empties) <= self.max_chance_branches:
            prob = 1.0 / len(empties)
            expected = 0.0
            for pos in empties:
                next_mat = _add_two_at(_clone(mat), pos)
                expected += prob * self._max_node(next_mat, depth - 1)
            return expected

        # 空位太多，需要选择代表性位置进行近似
        if self.sample_chance:
            # 采样模式：随机选择 k 个位置，用平均值近似期望
            k = min(self.max_chance_branches, len(empties))
            picks = random.sample(empties, k)
            total = 0.0
            for pos in picks:
                next_mat = _add_two_at(_clone(mat), pos)
                total += self._max_node(next_mat, depth - 1)
            return total / k
        else:
            # 智能选择模式：优先考虑危险位置（更保守的策略）
            # 在 Expectimax 中，我们倾向于评估"更坏"的情况，以获得更保守的估计
            dangerous_positions = self._get_dangerous_positions(mat, empties)
            
            k = self.max_chance_branches
            if len(dangerous_positions) >= k:
                # 选择最危险的 k 个位置（这些位置出现新 tile 通常更不利）
                candidates = dangerous_positions[:k]
            else:
                # 包含所有危险位置，再随机补充其他位置以保证多样性
                candidates = dangerous_positions[:]
                remaining = [pos for pos in empties if pos not in candidates]
                if remaining:
                    k_remaining = k - len(candidates)
                    candidates.extend(random.sample(remaining, min(k_remaining, len(remaining))))
            
            # 计算这些候选位置的平均得分作为期望值的近似
            # 由于我们优先选择了危险位置，这个估计会偏向保守（更低的期望值）
            total = 0.0
            for pos in candidates:
                next_mat = _add_two_at(_clone(mat), pos)
                total += self._max_node(next_mat, depth - 1)
            
            return total / len(candidates)
