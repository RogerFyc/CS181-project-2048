# agent_minimax.py
import math
import random
from functools import lru_cache
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
    """在指定空位放入2（你的版本只会生成2）。"""
    i, j = pos
    mat[i][j] = 2
    return mat


def _log2(v):
    return 0 if v <= 0 else int(math.log2(v))


def _heuristic(mat):
    """
    评估函数：越大越好（站在玩家角度）。
    典型要素：空格多、最大块大、局面更“平滑”、更“单调”。
    """
    n = len(mat)
    logs = [[_log2(mat[i][j]) for j in range(n)] for i in range(n)]

    empty = sum(1 for i in range(n) for j in range(n) if mat[i][j] == 0)
    max_tile = max(max(row) for row in mat)

    # smoothness：相邻差越小越好
    smooth = 0
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                smooth -= abs(logs[i][j] - logs[i + 1][j])
            if j + 1 < n:
                smooth -= abs(logs[i][j] - logs[i][j + 1])

    # monotonicity：鼓励行列整体单调（更容易继续合并）
    mono = 0
    # rows
    for i in range(n):
        for j in range(n - 1):
            mono += logs[i][j] - logs[i][j + 1] if logs[i][j] >= logs[i][j + 1] else -(logs[i][j + 1] - logs[i][j])
    # cols
    for j in range(n):
        for i in range(n - 1):
            mono += logs[i][j] - logs[i + 1][j] if logs[i][j] >= logs[i + 1][j] else -(logs[i + 1][j] - logs[i][j])

    # 权重（你可以后续调参）
    return (
        empty * 500
        + _log2(max_tile) * 200
        + smooth * 100
        + mono * 100
    )


class MinimaxAgent:
    """
    Minimax（对抗最坏情况）：
    - Max层：玩家选方向
    - Min层：对手选“2”出现的位置（最坏落点）
    """

    def __init__(self, depth=3, special_pos=None, max_min_branches=6):
        self.depth = depth
        self.special_pos = special_pos
        self.max_min_branches = max_min_branches

    def choose_move(self, mat):
        """返回 'Up'/'Down'/'Left'/'Right' 或 None(无路可走)。"""
        best_move = None
        best_score = -float("inf")

        for move_name, move_fn in MOVE_FUNCS.items():
            next_mat, done = move_fn(_clone(mat))
            if not done:
                continue

            # 改版规则：先应用特殊格子减半
            next_mat = _apply_special_cell_effect(next_mat, self.special_pos)

            score = self._min_node(next_mat, self.depth - 1)
            if score > best_score:
                best_score = score
                best_move = move_name

        return best_move

    def _min_node(self, mat, depth):
        """
        Min层:选择最坏的 tile 出现位置(只会出现2)。
        """
        state = logic.game_state(mat)
        if depth <= 0 or state != "not over":
            return _heuristic(mat)

        empties = _empty_cells(mat)
        if not empties:
            return _heuristic(mat)

        # 分支太多会爆炸：只选一部分“更可能更坏”的落点评估
        candidates = empties
        if len(empties) > self.max_min_branches:
            # 先粗评一下每个落点的启发式分数，取最差的若干个
            scored = []
            for pos in empties:
                tmp = _add_two_at(_clone(mat), pos)
                scored.append((_heuristic(tmp), pos))
            scored.sort(key=lambda x: x[0])  # 越小越坏
            candidates = [pos for _, pos in scored[: self.max_min_branches]]

        worst = float("inf")
        for pos in candidates:
            next_mat = _add_two_at(_clone(mat), pos)
            worst = min(worst, self._max_node(next_mat, depth - 1))
        return worst

    def _max_node(self, mat, depth):
        """
        Max层:玩家选方向。
        """
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
            best = max(best, self._min_node(next_mat, depth - 1))

        if not moved:
            return _heuristic(mat)

        return best
