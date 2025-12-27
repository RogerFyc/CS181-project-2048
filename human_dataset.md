# 2048 数据采集说明

## 快速开始

```bash
python puzzle. py
```

## 采集操作

- **C 键**：开启/关闭数据采集
- **V 键**：保存采集的数据

## 采集流程

```
1. 启动游戏
2. 切换到 Human 模式（H 或 M 键）
3. 按 C 开启数据采集
4. 玩游戏
5. 游戏结束后按 V 保存
```

## 保存的字段

### 局级别字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `episode` | int | 局编号 |
| `num_steps` | int | 步数 |
| `game_score` | float | 最终得分 |
| `game_steps` | int | 游戏步数 |
| `game_state` | str | "win" 或 "lose" |
| `data` | list | 该局所有步骤 |

### 步级别字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `step` | int | 步序号 |
| `state` | list | 动作前的棋盘状态（16维向量） |
| `action` | str | "Up"/"Down"/"Left"/"Right" |
| `reward` | float | 该步得到的分数 |
| `special_pos` | list | 特殊格子位置 [行, 列] |
| `next_state` | list | 动作后的棋盘状态（16维向量） |
| `done` | bool | 是否游戏结束 |

## 数据示例

```json
{
  "episode": 1,
  "num_steps": 3,
  "game_score":  20,
  "game_steps": 3,
  "game_state":  "lose",
  "data": [
    {
      "step": 0,
      "state": [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
      "action": "Up",
      "reward": 0.0,
      "special_pos": [1, 2],
      "next_state": [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
      "done": false
    }
  ]
}
```

## 矩阵展平

4×4 棋盘展平为 16 维向量（行优先）：

```
[a, b, c, d]
[e, f, g, h]    →    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]
[i, j, k, l]
[m, n, o, p]
```

## 统计文件

保存时会生成 `*_stats.json`，包含总局数、总步数、时间戳等统计信息。