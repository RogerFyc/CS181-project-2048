# DQN 奖励函数优化说明

## 🔍 问题分析：为什么平均奖励是负数？

### 原始奖励函数的问题：

1. **大多数步骤奖励为 0**
   - 只有发生合并时才有奖励（通常是 0-20）
   - 大多数步骤没有合并，奖励为 0

2. **失败惩罚过大**
   - 游戏失败时：`-1000`（严重惩罚）
   - 这个惩罚会严重影响平均奖励

3. **缺少正面激励**
   - 没有每步的基础奖励
   - 没有空格奖励（保持空格很重要）
   - 没有最大块奖励（鼓励创造更大的数字）

4. **结果**
   - 平均奖励 = (少量合并奖励 + 大量0 + 偶尔-100 + 失败时-1000) / 总步数
   - 由于失败惩罚很大，平均奖励通常是负数

---

## ✅ 优化后的奖励函数

### 新增的奖励项：

1. **每步基础奖励** (`step_reward = 1.0`)
   - 每步都有小的正奖励
   - 鼓励 agent 持续游戏，而不是快速失败

2. **空格奖励** (`empty_reward_scale = 2.0`)
   - 奖励 = 空格数量 × 2.0
   - 鼓励保持空格，避免填满棋盘

3. **最大块奖励** (`max_tile_reward_scale = 5.0`)
   - 奖励 = log2(最大块) × 5.0
   - 鼓励创造更大的数字

4. **改进的合并奖励** (`merge_reward_scale = 20`)
   - 基于合并产生的实际价值，而不仅仅是数量
   - 合并奖励从 10 增加到 20

5. **减少失败惩罚** (`terminal_penalty = -500`)
   - 从 -1000 减少到 -500
   - 平衡正面奖励和负面惩罚

---

## 📊 奖励函数组成

### 每步奖励计算：

```python
reward = 0.0

# 1. 每步基础奖励
reward += 1.0

# 2. 合并奖励
merge_value = calculate_merge_value(state, next_state)
reward += merge_value * 20

# 3. 空格奖励
empty_count = count_empty_cells(next_state)
reward += empty_count * 2.0

# 4. 最大块奖励
max_tile = get_max_tile(next_state)
reward += log2(max_tile) * 5.0

# 5. 特殊格惩罚/奖励
special_penalty = calculate_special_penalty(state, next_state)
reward += special_penalty

# 6. 游戏结束惩罚/奖励
if done:
    if lose:
        reward += -500
    elif win:
        reward += 1000
```

---

## 📈 预期效果

### 优化前（平均奖励为负）：
- 大多数步骤：奖励 = 0
- 偶尔合并：奖励 = +10 到 +20
- 偶尔大数字进入特殊格：奖励 = -100
- 游戏失败：奖励 = -1000
- **平均奖励：负数** ❌

### 优化后（平均奖励为正）：
- 每步：奖励 = +1（基础）
- 空格奖励：+2 到 +32（取决于空格数量）
- 最大块奖励：+5 到 +60（取决于最大块）
- 合并奖励：+20 到 +200（取决于合并价值）
- 游戏失败：奖励 = -500（减少）
- **平均奖励：正数** ✅

---

## 🎯 奖励函数参数说明

### 可调参数（在 `DQNAgent.__init__` 中）：

```python
self.step_reward = 1.0          # 每步基础奖励
self.empty_reward_scale = 2.0   # 空格奖励缩放
self.max_tile_reward_scale = 5.0  # 最大块奖励缩放
self.merge_reward_scale = 20    # 合并奖励缩放
self.terminal_penalty = -500    # 游戏结束惩罚
self.large_penalty = -100       # 大数字进入特殊格惩罚
self.small_reward = 10          # 小数字进入特殊格奖励
```

### 参数调整建议：

1. **如果平均奖励仍然为负**：
   - 增加 `step_reward`（如 2.0）
   - 增加 `empty_reward_scale`（如 3.0）
   - 减少 `terminal_penalty`（如 -300）

2. **如果平均奖励太高（失去区分度）**：
   - 减少 `step_reward`（如 0.5）
   - 减少 `empty_reward_scale`（如 1.0）

3. **如果 agent 不够积极**：
   - 增加 `merge_reward_scale`（如 30）
   - 增加 `max_tile_reward_scale`（如 10）

4. **如果 agent 太冒险**：
   - 增加 `large_penalty`（如 -150）
   - 增加 `terminal_penalty`（如 -700）

---

## 🔧 如何验证改进

### 1. 重新训练模型：

```bash
python train_dqn.py --episodes 500
```

### 2. 观察训练输出：

```
Episode 100 | Avg Reward: 50.23 | Avg Length: 120.5
Episode 200 | Avg Reward: 65.78 | Avg Length: 145.2
Episode 300 | Avg Reward: 78.45 | Avg Length: 160.8
...
```

**现在平均奖励应该是正数！** ✅

### 3. 在游戏中测试：

```bash
python puzzle.py
```

选择 DQN agent，观察：
- Agent 是否能持续游戏更长时间？
- 是否能达到更高的分数？
- 是否能更好地避免特殊格？

---

## 📝 奖励函数设计原则

1. **平衡正面和负面奖励**
   - 正面奖励应该足够大，让 agent 有动力
   - 负面惩罚应该足够大，让 agent 避免不良行为
   - 但总体应该是正数，鼓励持续游戏

2. **奖励应该与目标一致**
   - 空格奖励：鼓励保持空格（避免填满）
   - 最大块奖励：鼓励创造更大的数字
   - 合并奖励：鼓励合并（提高分数）

3. **避免奖励稀疏**
   - 每步都应该有奖励（即使是小的）
   - 避免只有特定事件才有奖励（如只有合并才有奖励）

4. **惩罚应该适度**
   - 失败惩罚不应该过大（否则会掩盖正面奖励）
   - 特殊格惩罚应该足够大（让 agent 避免）

---

## 🎉 总结

**优化后的奖励函数应该：**
- ✅ 平均奖励为正数
- ✅ 鼓励持续游戏
- ✅ 鼓励保持空格
- ✅ 鼓励创造更大的数字
- ✅ 鼓励合并
- ✅ 惩罚进入特殊格的大数字
- ✅ 适度惩罚游戏失败

**如果平均奖励仍然是负数，可以调整参数（见上面的参数调整建议）。**

