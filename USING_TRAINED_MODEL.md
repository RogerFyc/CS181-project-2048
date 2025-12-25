# 使用训练好的 DQN Agent 指南

## ✅ 好消息：你的模型已经准备好了！

我看到你已经有了训练好的模型文件：`dqn_2048_model.pth`

## 🎮 方法 1：在游戏中直接使用（最简单）

### 步骤：

1. **运行游戏：**
   ```bash
   python puzzle.py
   ```

2. **在游戏界面中：**
   - 点击 **"AI"** 单选按钮（切换到 AI 模式）
   - 点击 **"DQN"** 单选按钮（选择 DQN agent）
   - 勾选 **"Auto-play"**（自动游戏）

3. **模型会自动加载！**

   游戏启动时，如果检测到 `dqn_2048_model.pth` 文件，会自动加载并显示：
   ```
   Loaded DQN model from dqn_2048_model.pth
   ```

4. **开始游戏：**
   - Agent 会自动开始游戏
   - 或者按 **Space** 键单步执行

### 键盘快捷键：

- **`m`** - 切换 Human/AI 模式
- **`t`** - 切换 AI 类型（Minimax/Expectimax/DQN）
- **`Space`** - AI 单步执行（当 Auto-play 关闭时）
- **`r`** - 重新开始游戏

---

## 🐍 方法 2：在 Python 代码中使用

### 基本使用：

```python
from agent_Qlearning import DQNAgent
import logic

# 创建 agent
agent = DQNAgent(
    special_pos=None,  # 或指定位置，如 (1, 1)
    auto_detect_special=True
)

# 加载训练好的模型
agent.load("dqn_2048_model.pth")

# 获取当前游戏状态
game_matrix = logic.new_game(4)  # 4x4 棋盘

# 让 agent 选择移动
move = agent.choose_move(game_matrix)
print(f"Agent chose: {move}")  # 输出: Up, Down, Left, 或 Right
```

### 完整游戏循环示例：

```python
from agent_Qlearning import DQNAgent
import logic
import constants as c

# 创建并加载 agent
agent = DQNAgent(special_pos=None, auto_detect_special=True)
agent.load("dqn_2048_model.pth")

# 初始化游戏
matrix = logic.new_game(c.GRID_LEN)
steps = 0

# 游戏循环
while logic.game_state(matrix) == "not over":
    # Agent 选择移动
    move = agent.choose_move(matrix)
    
    if move is None:
        print("No valid moves!")
        break
    
    # 执行移动
    if move == "Up":
        matrix, _ = logic.up(matrix)
    elif move == "Down":
        matrix, _ = logic.down(matrix)
    elif move == "Left":
        matrix, _ = logic.left(matrix)
    elif move == "Right":
        matrix, _ = logic.right(matrix)
    
    # 应用特殊格效果（如果需要）
    # ... 你的特殊格逻辑 ...
    
    # 添加新 tile
    matrix = logic.add_two(matrix)
    
    steps += 1
    print(f"Step {steps}: {move}")

print(f"Game over after {steps} steps")
```

---

## 🔍 验证模型是否加载成功

### 在游戏中验证：

1. 运行游戏：`python puzzle.py`
2. 选择 DQN agent
3. 查看控制台输出，应该看到：
   ```
   Loaded DQN model from dqn_2048_model.pth
   ```

### 在代码中验证：

```python
from agent_Qlearning import DQNAgent

agent = DQNAgent()
try:
    agent.load("dqn_2048_model.pth")
    print("✅ Model loaded successfully!")
    print(f"Current epsilon: {agent.epsilon}")  # 应该接近 0.01（训练结束时的值）
except Exception as e:
    print(f"❌ Failed to load model: {e}")
```

---

## 📊 测试模型性能

### 在游戏中测试：

1. 运行游戏并选择 DQN agent
2. 观察 agent 的表现：
   - 是否能达到较高的分数？
   - 是否能避免特殊格？
   - 游戏长度如何？

### 批量测试（代码）：

```python
from agent_Qlearning import DQNAgent
import logic
import constants as c

agent = DQNAgent(special_pos=None, auto_detect_special=True)
agent.load("dqn_2048_model.pth")

# 测试多局游戏
num_games = 10
results = []

for game in range(num_games):
    matrix = logic.new_game(c.GRID_LEN)
    steps = 0
    max_tile = 0
    
    while logic.game_state(matrix) == "not over":
        move = agent.choose_move(matrix)
        if move is None:
            break
        
        # 执行移动（简化版）
        if move == "Up":
            matrix, _ = logic.up(matrix)
        elif move == "Down":
            matrix, _ = logic.down(matrix)
        elif move == "Left":
            matrix, _ = logic.left(matrix)
        elif move == "Right":
            matrix, _ = logic.right(matrix)
        
        matrix = logic.add_two(matrix)
        steps += 1
        max_tile = max(max(row) for row in matrix)
    
    results.append({
        'steps': steps,
        'max_tile': max_tile,
        'state': logic.game_state(matrix)
    })
    print(f"Game {game+1}: {steps} steps, max tile: {max_tile}")

# 统计结果
avg_steps = sum(r['steps'] for r in results) / len(results)
avg_max_tile = sum(r['max_tile'] for r in results) / len(results)
print(f"\nAverage: {avg_steps:.1f} steps, {avg_max_tile:.1f} max tile")
```

---

## ⚙️ 使用不同名称的模型文件

如果你的模型文件名不是 `dqn_2048_model.pth`，有两种方法：

### 方法 1：重命名文件

```bash
# 将你的模型文件重命名为默认名称
mv your_model.pth dqn_2048_model.pth
```

### 方法 2：修改代码（临时）

在 `puzzle.py` 中修改模型路径（第 128 行）：

```python
model_path = "your_model.pth"  # 改为你的文件名
```

### 方法 3：在代码中手动加载

```python
from agent_Qlearning import DQNAgent

agent = DQNAgent()
agent.load("your_model.pth")  # 使用你的模型文件名
```

---

## 🎯 最佳实践

### 1. 确保模型文件在正确位置

模型文件 `dqn_2048_model.pth` 应该和 `puzzle.py` 在同一目录下。

### 2. 检查模型是否训练充分

- 训练 500 次：可能性能一般，适合测试
- 训练 10000+ 次：性能更好，适合实际使用

### 3. 观察 agent 行为

- **好的表现**：能持续游戏，避免特殊格，达到较高分数
- **需要改进**：频繁失败，无法避免特殊格，分数较低

### 4. 继续训练（如果需要）

如果模型性能不够好，可以继续训练：

```bash
# 从现有模型继续训练
python train_dqn.py \
    --load-path dqn_2048_model.pth \
    --episodes 5000 \
    --save-path dqn_2048_model_v2.pth
```

---

## 🐛 常见问题

### Q1: 模型加载失败？

**检查：**
- 模型文件是否存在？
- 文件路径是否正确？
- PyTorch 版本是否兼容？

**解决方案：**
```python
import torch
print(torch.__version__)  # 检查 PyTorch 版本
```

### Q2: Agent 表现很差？

**可能原因：**
- 训练次数太少（500 次可能不够）
- 模型文件损坏
- 特殊格位置不匹配

**解决方案：**
- 继续训练更多 episodes
- 重新训练模型
- 检查特殊格位置设置

### Q3: 如何知道模型是否在运行？

**检查方法：**
- 查看控制台输出（应该显示 "Loaded DQN model..."）
- 观察游戏中的移动（应该是有策略的，不是完全随机）
- 检查 agent 的 epsilon 值（应该接近 0.01）

---

## 📝 快速开始清单

- [ ] 确认 `dqn_2048_model.pth` 文件存在
- [ ] 运行 `python puzzle.py`
- [ ] 选择 "AI" 模式
- [ ] 选择 "DQN" agent
- [ ] 勾选 "Auto-play"
- [ ] 观察 agent 表现

---

## 🎉 总结

**最简单的使用方法：**

1. 运行游戏：`python puzzle.py`
2. 选择 DQN agent
3. 开始游戏！

模型会自动加载，无需额外配置！🚀



