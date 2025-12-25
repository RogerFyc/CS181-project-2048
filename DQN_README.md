# Deep Q-Learning (DQN) Agent 使用说明

## 安装依赖

DQN agent 需要 PyTorch：

```bash
pip install torch
```

## 快速开始

### 1. 训练 DQN Agent

```python
from agent_Qlearning import DQNAgent, train_dqn_agent

# 创建 agent（特殊格位置需要与游戏一致）
special_pos = (1, 1)  # 示例：特殊格在中心
agent = DQNAgent(
    special_pos=special_pos,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    memory_size=100000,
    batch_size=64,
    target_update_freq=1000
)

# 开始训练
train_dqn_agent(
    agent, 
    num_episodes=10000, 
    save_freq=1000, 
    save_path="dqn_2048_model.pth"
)
```

### 2. 使用已训练的模型

在 `puzzle.py` 中：
1. 选择 AI Type 为 "DQN"
2. Agent 会自动尝试加载 `dqn_2048_model.pth`
3. 如果模型不存在，会使用未训练的 agent

### 3. 手动加载模型

```python
from agent_Qlearning import DQNAgent

agent = DQNAgent(special_pos=(1, 1))
agent.load("dqn_2048_model.pth")

# 使用 agent 选择动作
move = agent.choose_move(game_matrix)
```

## 架构说明

### 状态表示
- **18维向量**：
  - 前16维：4×4棋盘的值（log2编码，空值为0）
  - 后2维：特殊格位置（归一化到[0,1]）

### 动作空间
- 4个动作：Up, Down, Left, Right

### 奖励函数
1. **合并奖励**：每次合并 +10 分
2. **特殊格惩罚**：
   - 大数字（≥32）进入特殊格：-100
   - 小数字（≤8）进入特殊格：+10
3. **游戏结束惩罚**：-1000（失败）或 +1000（胜利）

### 网络结构
- 输入层：18维
- 隐藏层1：256维（ReLU）
- 隐藏层2：256维（ReLU）
- 隐藏层3：256维（ReLU）
- 输出层：4维（每个动作的Q值）

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 0.001 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `epsilon_start` | 1.0 | 初始探索率 |
| `epsilon_end` | 0.01 | 最终探索率 |
| `epsilon_decay` | 0.995 | 探索率衰减 |
| `memory_size` | 100000 | 经验回放缓冲区大小 |
| `batch_size` | 64 | 批次大小 |
| `target_update_freq` | 1000 | 目标网络更新频率 |

## 训练建议

1. **初始训练**：10000-50000 episodes
2. **观察指标**：
   - 平均奖励（应该逐渐增加）
   - 平均游戏长度（应该逐渐增加）
   - Epsilon值（应该逐渐降低）
3. **调整策略**：
   - 如果奖励不增加：降低学习率或增加网络容量
   - 如果训练不稳定：增加目标网络更新频率
   - 如果探索不足：调整 epsilon 衰减率

## 注意事项

1. **特殊格位置**：训练时和测试时的特殊格位置必须一致
2. **模型保存**：训练过程中会定期保存模型
3. **GPU加速**：如果有GPU，会自动使用（需要安装CUDA版本的PyTorch）
4. **内存使用**：经验回放缓冲区会占用内存，可根据需要调整 `memory_size`

## 故障排除

### PyTorch 未安装
```
ImportError: PyTorch is required for DQN agent
```
解决：`pip install torch`

### 模型加载失败
- 检查模型文件路径是否正确
- 确保模型文件完整
- 检查特殊格位置是否匹配

### 训练不收敛
- 尝试降低学习率
- 增加训练轮数
- 调整奖励函数参数
- 检查网络结构是否合适




