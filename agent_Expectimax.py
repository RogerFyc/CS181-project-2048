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


def _heuristic(mat):
    """
    启发式评估：越大越好（站在玩家角度）。
    这里沿用你 Minimax 版本的风格：空格、最大块、平滑、单调。
    """
    n = len(mat)
    logs = [[_log2(mat[i][j]) for j in range(n)] for i in range(n)]

    empty = sum(1 for i in range(n) for j in range(n) if mat[i][j] == 0)
    max_tile = max(max(row) for row in mat)

    smooth = 0
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                smooth -= abs(logs[i][j] - logs[i + 1][j])
            if j + 1 < n:
                smooth -= abs(logs[i][j] - logs[i][j + 1])

    mono = 0
    for i in range(n):
        for j in range(n - 1):
            mono += logs[i][j] - logs[i][j + 1] if logs[i][j] >= logs[i][j + 1] else -(logs[i][j + 1] - logs[i][j])
    for j in range(n):
        for i in range(n - 1):
            mono += logs[i][j] - logs[i + 1][j] if logs[i][j] >= logs[i + 1][j] else -(logs[i + 1][j] - logs[i][j])

    return (
        empty * 1000
        + _log2(max_tile) * 200
        + smooth * 5
        + mono * 2
    )


class ExpectimaxAgent:
    """
    Expectimax：
    - Max层：玩家选方向（最大化期望得分）
    - Chance层：随机生成新 tile 的位置（你这版只生成2，因此只对“位置”做期望）
    """

    def __init__(self, depth=3, special_pos=None, max_chance_branches=10, sample_chance=False):
        """
        depth: 搜索深度
        special_pos: 特殊格子位置（用于模拟减半规则）
        max_chance_branches: 空位太多时，为了速度限制 chance 分支数
        sample_chance: True 时用采样近似期望（更快但有随机性）
        """
        self.depth = depth
        self.special_pos = special_pos
        self.max_chance_branches = max_chance_branches
        self.sample_chance = sample_chance

    def choose_move(self, mat):
        """返回 'Up'/'Down'/'Left'/'Right' 或 None（无路可走）。"""
        best_move = None
        best_score = -float("inf")

        for move_name, move_fn in MOVE_FUNCS.items():
            next_mat, done = move_fn(_clone(mat))
            if not done:
                continue

            # 改版规则：先应用特殊格子减半
            next_mat = _apply_special_cell_effect(next_mat, self.special_pos)

            score = self._chance_node(next_mat, self.depth - 1)
            if score > best_score:
                best_score = score
                best_move = move_name

        return best_move

    def _max_node(self, mat, depth):
        """Max层：玩家选方向。"""
        state = logic.game_state(mat)
        if depth <= 0 or state != "not over":
            return _heuristic(mat)

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
            return _heuristic(mat)
        return best

    def _chance_node(self, mat, depth):
        """
        Chance层：随机生成新 tile 的位置。
        你这版只生成2，所以概率 = 1/空位数（均匀）。
        """
        state = logic.game_state(mat)
        if depth <= 0 or state != "not over":
            return _heuristic(mat)

        empties = _empty_cells(mat)
        if not empties:
            return _heuristic(mat)

        # 分支控制：空位过多时可采样或截断
        if self.sample_chance and len(empties) > self.max_chance_branches:
            # 采样近似期望
            k = self.max_chance_branches
            picks = random.sample(empties, k)
            total = 0.0
            for pos in picks:
                next_mat = _add_two_at(_clone(mat), pos)
                total += self._max_node(next_mat, depth - 1)
            return total / k

        # 截断但尽量“代表性”：随机选一部分位置做平均
        candidates = empties
        if len(empties) > self.max_chance_branches:
            candidates = random.sample(empties, self.max_chance_branches)

        prob = 1.0 / len(candidates)
        expected = 0.0
        for pos in candidates:
            next_mat = _add_two_at(_clone(mat), pos)
            expected += prob * self._max_node(next_mat, depth - 1)
        return expected
