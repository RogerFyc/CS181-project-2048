# DQN Agent 训练指南

## 📋 快速开始

### 方法 1：使用训练脚本（推荐）

```bash
# 基础训练（10000 episodes）
python Agents/Agents/train_dqn.py

# 自定义参数训练
python Agents/Agents/train_dqn.py --episodes 20000 --save-freq 2000 --learning-rate 0.0005
```

### 方法 2：在 Python 中直接训练

```python
from Agents.agent_Qlearning import DQNAgent, train_dqn_agent

# 创建 agent
agent = DQNAgent(
    special_pos=None,  # 自动检测特殊格位置
    auto_detect_special=True,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995
)

# 开始训练
train_dqn_agent(agent, num_episodes=10000, save_path="Data/dqn_2048_model.pth")
```

---

## 🎯 训练参数说明

### 基础参数

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `num_episodes` | 10000 | 训练轮数 | 5000-50000 |
| `save_freq` | 1000 | 保存频率 | 500-2000 |
| `learning_rate` | 0.001 | 学习率 | 0.0001-0.01 |
| `gamma` | 0.99 | 折扣因子 | 0.95-0.99 |
| `epsilon_start` | 1.0 | 初始探索率 | 1.0 |
| `epsilon_end` | 0.01 | 最终探索率 | 0.01-0.1 |
| `epsilon_decay` | 0.995 | 探索率衰减 | 0.99-0.999 |
| `batch_size` | 64 | 批次大小 | 32-128 |
| `memory_size` | 100000 | 经验回放缓冲区大小 | 50000-200000 |
| `target_update_freq` | 1000 | 目标网络更新频率 | 500-2000 |

### 特殊格位置

- **自动检测**（推荐）：`special_pos=None, auto_detect_special=True`
- **手动指定**：`special_pos=(1, 1), auto_detect_special=False`

---

## 📊 训练过程监控

训练过程中会每 100 episodes 打印一次统计信息：

```
Episode 100/10000 | Avg Reward: -1234.56 | Avg Length: 45.23 | Epsilon: 0.951
```

### 关键指标

- **Avg Reward**：平均奖励（应该逐渐增加）
- **Avg Length**：平均游戏长度（应该逐渐增加）
- **Epsilon**：当前探索率（应该逐渐降低）

### 正常训练表现

- ✅ 平均奖励逐渐增加（从负数变为正数）
- ✅ 平均游戏长度逐渐增加
- ✅ Epsilon 逐渐降低（探索减少，利用增加）

### 异常情况

- ❌ 平均奖励一直为负数且不增加 → 降低学习率或调整奖励函数
- ❌ 平均长度很短 → 检查奖励函数，可能需要增加空格奖励
- ❌ 训练不稳定（奖励波动大） → 增加目标网络更新频率

---

## 🔧 训练配置建议

### 快速测试（验证代码）

```bash
python Agents/Agents/train_dqn.py --episodes 1000 --save-freq 500
```

### 基础训练（推荐开始）

```bash
python Agents/Agents/train_dqn.py --episodes 10000 --save-freq 1000
```

### 深度训练（追求更好性能）

```bash
python Agents/Agents/train_dqn.py \
    --episodes 50000 \
    --save-freq 2000 \
    --learning-rate 0.0005 \
    --epsilon-decay 0.998 \
    --batch-size 128
```

### 继续训练（从已有模型继续）

```bash
python Agents/Agents/train_dqn.py \
    --episodes 20000 \
    --load-path Data/dqn_2048_model.pth \
    --save-path Data/dqn_2048_model_v2.pth
```

---

## 💡 训练技巧

### 1. 渐进式训练

```bash
# 第一阶段：快速探索（高探索率）
python Agents/Agents/train_dqn.py --episodes 5000 --epsilon-decay 0.99

# 第二阶段：精细调优（低探索率）
python Agents/Agents/train_dqn.py --episodes 20000 --load-path Data/dqn_2048_model.pth --epsilon-start 0.1 --epsilon-decay 0.999
```

### 2. 调整奖励函数

如果训练效果不好，可以在 `agent_Qlearning.py` 中调整：

```python
# 在 DQNAgent.__init__ 中
self.merge_reward_scale = 10      # 合并奖励缩放
self.large_penalty = -100         # 大数字进入特殊格惩罚
self.small_reward = 10             # 小数字进入特殊格奖励
self.terminal_penalty = -1000      # 游戏结束惩罚
```

### 3. 使用 GPU 加速

如果有 NVIDIA GPU：

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Agent 会自动使用 GPU（如果可用）
```

### 4. 监控训练进度

训练过程中可以：
- 观察控制台输出的统计信息
- 定期检查保存的模型文件
- 在游戏中测试已训练的模型

---

## 📈 训练示例

### 完整训练流程

```bash
# 1. 初始训练（10000 episodes）
python Agents/Agents/train_dqn.py --episodes 10000 --save-path Data/dqn_2048_model.pth

# 2. 测试模型性能
# 在游戏中加载模型并测试

# 3. 继续训练（如果效果不够好）
python Agents/Agents/train_dqn.py \
    --episodes 20000 \
    --load-path Data/dqn_2048_model.pth \
    --save-path Data/dqn_2048_model_v2.pth \
    --learning-rate 0.0005

# 4. 最终优化
python Agents/Agents/train_dqn.py \
    --episodes 20000 \
    --load-path dqn_2048_model_v2.pth \
    --save-path Data/dqn_2048_model_final.pth \
    --epsilon-start 0.1 \
    --epsilon-decay 0.9995
```

---

## ⚠️ 常见问题

### Q1: 训练很慢怎么办？

**解决方案：**
- 减少 `num_episodes`（先测试 1000 episodes）
- 减少 `memory_size`（如果内存不足）
- 减少 `batch_size`（如果显存不足）
- 使用 GPU（如果有）

### Q2: 训练不收敛怎么办？

**解决方案：**
- 降低学习率（`--learning-rate 0.0005`）
- 增加目标网络更新频率（`--target-update-freq 2000`）
- 调整奖励函数参数
- 增加训练轮数

### Q3: 内存不足怎么办？

**解决方案：**
- 减少 `memory_size`（`--memory-size 50000`）
- 减少 `batch_size`（`--batch-size 32`）

### Q4: 如何中断训练？

**按 `Ctrl+C`**，模型会自动保存到指定路径。

### Q5: 训练好的模型在哪里？

模型保存在当前目录下的 `Data/dqn_2048_model.pth`（或你指定的路径）。

---

## 🎮 使用训练好的模型

### 在游戏中加载

1. 运行游戏：`python puzzle.py`
2. 选择 AI Type 为 "DQN"
3. Agent 会自动加载 `Data/dqn_2048_model.pth`（如果存在）

### 手动加载

```python
from Agents.agent_Qlearning import DQNAgent

agent = DQNAgent(special_pos=(1, 1))
agent.load("Data/dqn_2048_model.pth")

# 使用 agent
move = agent.choose_move(game_matrix)
```

---

## 📝 训练日志示例

```
============================================================
DQN Agent Training for 2048 Game
============================================================
Special tile position: Auto-detect enabled

Creating DQN Agent...
Starting training from scratch...

Training parameters:
  Episodes: 10000
  Save frequency: 1000
  Save path: Data/dqn_2048_model.pth
  Learning rate: 0.001
  Gamma: 0.99
  Batch size: 64
  Device: cuda
============================================================
Episode 100/10000 | Avg Reward: -1234.56 | Avg Length: 45.23 | Epsilon: 0.951
Episode 200/10000 | Avg Reward: -987.65 | Avg Length: 52.34 | Epsilon: 0.904
...
Episode 10000/10000 | Avg Reward: 1234.56 | Avg Length: 234.56 | Epsilon: 0.010
============================================================
Training completed successfully!
Final model saved to: Data/dqn_2048_model.pth
Average reward (last 100 episodes): 1234.56
Average length (last 100 episodes): 234.56
============================================================
```

---

## 🚀 快速开始命令

```bash
# 最简单的训练（使用默认参数）
python Agents/Agents/train_dqn.py

# 快速测试（1000 episodes）
python Agents/Agents/train_dqn.py --episodes 1000

# 完整训练（20000 episodes）
python Agents/Agents/train_dqn.py --episodes 20000 --save-freq 2000
```

---

## 📚 更多信息

- 查看 `DQN_README.md` 了解 DQN agent 的详细说明
- 查看 `agent_Qlearning.py` 了解实现细节
- 在游戏中测试训练好的模型







> 也可以使用模块方式运行：`python -m Agents.train_dqn ...`（更推荐，路径更稳）。
