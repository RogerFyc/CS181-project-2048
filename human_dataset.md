# RL 训练数据集使用说明（Human/AI 采集）

本说明仅针对「2048 人类/AI 采集数据集」的使用方法与参数，帮助组员将 JSON 数据集转换为 CSV，并在 RL 训练中使用。

---

## 1. 数据来源与结构

采集脚本会把每一局的步骤数据保存为 JSON（列表形式），位于 `collected_data/` 目录。典型字段如下：

- 局级（episode）：
  - `episode`: 局编号
  - `num_steps`: 步数
  - `game_score`: 最终分数（棋盘所有值的和）
  - `game_steps`: 游戏步数（同 `num_steps`）
  - `game_state`: `"win"` 或 `"lose"`
  - `data`: 步骤列表

- 步级（step）（用于 RL 训练的状态转移）：
  - `step`: 步序号
  - `state`: 动作前棋盘状态（展平 16 维向量，行优先）
  - `action`: `"Up" | "Down" | "Left" | "Right"`
  - `special_pos`: 特殊格位置 `[row, col]` 或 `null`
  - `next_state`: 动作后的棋盘状态（展平 16 维向量）
  - `empty_cells`: 动作后空格数量
  - `times_max_reduced`: 迄今为止最大块被减半的累计次数
  - `times_other_reduced`: 迄今为止其他块被减半的累计次数
  - `max_tile_val`: 动作后棋盘的最大值

> 注意：`reward`、`done` 不在当前版本的采集数据中保存（奖励通常由训练过程定义；终止由 `game_state`/步序推断）。

---

## 2. 将每个 JSON 数据集导出为 CSV

脚本：`export_all_datasets_to_csv.py`

- 功能：把 `collected_data/` 目录下每一个 JSON 文件，分别导出为对应的 CSV 文件。
- 输出：
  - `{base}_steps.csv`: 步骤级数据（RL 训练推荐）
  - `{base}_episodes.csv`: 局级摘要（可选）

### 快速使用

```bash
# 导出所有数据集的 steps CSV（推荐）
python export_all_datasets_to_csv.py

# 指定目录，且只导出 steps CSV
python export_all_datasets_to_csv.py --input-dir collected_data --steps-only

# 仅导出 episodes 摘要 CSV
python export_all_datasets_to_csv.py --episodes-only

# 只处理某一个文件
python export_all_datasets_to_csv.py --file collected_data/2048_human_data_10games_20251228_162430.json

# 在每个数据文件内重写 episode ID 为 1..N
python export_all_datasets_to_csv.py --reindex
```

### 步骤级 CSV 列说明（`*_steps.csv`）

- `episode`: 局编号
- `step`: 步序号
- `action`: `"Up" | "Down" | "Left" | "Right"`
- `empty_cells`: 动作后空格数
- `times_max_reduced`: 截止到该步最大块被减半的累计次数
- `times_other_reduced`: 截止到该步非最大块被减半的累计次数
- `max_tile_val`: 动作后最大块值
- `special_pos_row`, `special_pos_col`: 特殊格位置行列（无则空）
- `state_0..state_15`: 动作前状态的 16 维展开
- `next_0..next_15`: 动作后状态的 16 维展开

### 局级摘要 CSV 列说明（`*_episodes.csv`）

- `episode`: 局编号
- `num_steps`: 步数（优先使用原字段，缺失时从步骤数推断）
- `game_score`: 最终分数（棋盘总和）
- `game_steps`: 游戏步数（同上）
- `game_state`: `"win" | "lose"`
- `max_tile`: 该局最大块（优先从步骤字段 `max_tile_val`，回退分析最后一步状态）
- `times_max_reduced`, `times_other_reduced`: 全局累计（取最后一步的累计值）

---

## 3. 数据在 RL 训练中的使用建议

### 3.1 状态表示

- 原始矩阵被展平为 16 维向量（`state_0..15`、`next_0..15`）。
- 数值为实际棋盘上的整数值（非 log2 编码）。若你的模型使用 log2 编码，需自行转换：
  ```python
  import math
  def encode_log2(x):
      return 0 if x <= 0 else int(math.log2(x))
  ```

### 3.2 动作映射

- `action` 为字符串方向；建议在训练前映射到离散动作索引：
  ```python
  ACTION_MAP = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
  ```

### 3.3 奖励与终止

- 奖励不直接来自数据集，建议在训练代码中定义（如基于合并、空格数、最大块、特殊格影响等）。
- 终止（`done`）可由步序与局级信息推断：
  - 若该步是 `episode` 的最后一步，下一步不存在，则 `done=True`。
  - 或使用 `*_episodes.csv` 的 `game_state` 辅助评估。

### 3.4 数据划分与采样

- 建议按局（`episode`）划分训练/验证集，避免同局步骤泄露到验证集。
- 经验回放（ReplayBuffer）可直接从 `*_steps.csv` 生成 `(s, a, r, s', done)`：
  ```python
  import csv

  ACTION_MAP = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}

  def load_steps_csv(path):
      transitions = []
      with open(path, newline='', encoding='utf-8') as f:
          reader = csv.DictReader(f)
          rows = list(reader)

      # 按 episode 分组，推断 done
      from collections import defaultdict
      episodes = defaultdict(list)
      for row in rows:
          episodes[int(row['episode'])].append(row)

      for ep_rows in episodes.values():
          ep_rows.sort(key=lambda r: int(r['step']))
          for i, r in enumerate(ep_rows):
              s = [int(r[f'state_{k}'] or 0) for k in range(16)]
              s_next = [int(r[f'next_{k}'] or 0) for k in range(16)]
              a = ACTION_MAP.get(r['action'], 0)
              done = (i == len(ep_rows) - 1)
              transitions.append((s, a, s_next, done))
      return transitions
  ```

---

## 4. 常见参数与流程

### 4.1 导出参数回顾

- `--input-dir`: 数据集所在目录（默认：`collected_data`）
- `--pattern`: 文件匹配模式（默认：`*.json`）
- `--file`: 仅处理某个特定 JSON 文件
- `--steps-only`: 只导出步骤级 CSV
- `--episodes-only`: 只导出摘要 CSV
- `--reindex`: 在每个数据文件内重排 episode ID 为 1..N

### 4.2 推荐流程（组员）

1. 采集数据（Human/AI，按 `C` 开始，按 `V` 保存到 `collected_data/`）
2. 导出 CSV：
   ```bash
   python export_all_datasets_to_csv.py --input-dir collected_data --steps-only
   ```
3. 在训练代码中从 `*_steps.csv` 加载 transitions，定义奖励，开始训练。

---

## 5. 注意事项

- 请跳过 `*_stats.json` 文件（脚本已自动忽略）。
- 如果需要统一全局 episode ID（跨多个数据文件），请先使用合并脚本 `merge_datasets.py`，再导出。
- 步骤 CSV 中未包含奖励与 `done`，是为了保持训练的奖励可配置、终止可推断的灵活性。

---