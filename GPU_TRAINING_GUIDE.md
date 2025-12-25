# GPU 训练指南

## ✅ 好消息：代码已自动支持 GPU！

DQN agent 已经内置了 GPU 支持，**会自动检测并使用 GPU**（如果可用）。

## 🚀 快速开始

### 1. 检查 GPU 是否可用

运行以下命令检查：

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
```

### 2. 自动使用 GPU

**无需任何额外配置！** 代码会自动检测并使用 GPU：

```python
from agent_Qlearning import DQNAgent

# 创建 agent（会自动使用 GPU，如果可用）
agent = DQNAgent(
    special_pos=None,
    auto_detect_special=True,
    learning_rate=0.001
)

# 查看使用的设备
print(f"Using device: {agent.device}")
# 输出: Using device: cuda 或 Using device: cpu
```

### 3. 训练时查看设备信息

运行训练脚本时，会自动显示使用的设备：

```bash
python train_dqn.py --episodes 10000
```

输出示例：
```
============================================================
DQN Agent Training for 2048 Game
============================================================
...
Training parameters:
  Device: cuda  # 或 cpu
============================================================
```

## 📦 安装 CUDA 版本的 PyTorch

### 如果还没有安装 PyTorch

**CPU 版本（默认）：**
```bash
pip install torch
```

**GPU 版本（推荐，如果 NVIDIA GPU）：**

1. **检查 CUDA 版本：**
   ```bash
   nvidia-smi
   ```
   查看 CUDA Version（例如：12.1）

2. **安装对应版本的 PyTorch：**

   **CUDA 11.8：**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   **CUDA 12.1：**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   **最新稳定版（推荐）：**
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **验证安装：**
   ```python
   import torch
   print(torch.cuda.is_available())  # 应该输出 True
   print(torch.cuda.get_device_name(0))  # 显示 GPU 名称
   ```

## 🎯 强制使用 CPU（如果需要）

如果你想强制使用 CPU（例如用于调试）：

```python
import torch
from agent_Qlearning import DQNAgent

# 强制使用 CPU
agent = DQNAgent(
    special_pos=None,
    device=torch.device("cpu"),  # 强制使用 CPU
    learning_rate=0.001
)
```

## 🎯 强制使用特定 GPU

如果你有多个 GPU，可以指定使用哪个：

```python
import torch
from agent_Qlearning import DQNAgent

# 使用第一个 GPU
agent = DQNAgent(
    special_pos=None,
    device=torch.device("cuda:0"),  # 使用 GPU 0
    learning_rate=0.001
)

# 使用第二个 GPU
agent = DQNAgent(
    special_pos=None,
    device=torch.device("cuda:1"),  # 使用 GPU 1
    learning_rate=0.001
)
```

## 📊 GPU vs CPU 性能对比

| 项目 | CPU | GPU |
|------|-----|-----|
| **训练速度** | 基准 | **5-20x 更快** |
| **批次大小** | 32-64 | **64-256** |
| **内存使用** | 较低 | 较高 |
| **适用场景** | 小规模训练 | **大规模训练** |

### 实际性能提升

- **小规模训练（1000 episodes）**：GPU 可能不明显
- **大规模训练（10000+ episodes）**：GPU 可以显著加速（5-20倍）

## 🔧 优化 GPU 使用

### 1. 增加批次大小

GPU 可以处理更大的批次：

```python
agent = DQNAgent(
    batch_size=128,  # GPU 可以使用更大的批次（CPU 建议 32-64）
    memory_size=200000,  # 也可以增加经验回放缓冲区
)
```

### 2. 调整训练参数

```bash
# GPU 训练建议参数
python train_dqn.py \
    --episodes 50000 \
    --batch-size 128 \
    --memory-size 200000 \
    --learning-rate 0.001
```

## ⚠️ 常见问题

### Q1: 为什么显示 "Device: cpu"？

**可能原因：**
1. 没有安装 CUDA 版本的 PyTorch
2. 没有 NVIDIA GPU
3. CUDA 驱动未安装或版本不匹配

**解决方案：**
- 检查 `nvidia-smi` 是否正常工作
- 安装 CUDA 版本的 PyTorch
- 检查 CUDA 版本兼容性

### Q2: GPU 内存不足怎么办？

**解决方案：**
```python
# 减少批次大小
agent = DQNAgent(
    batch_size=32,  # 从 64 减少到 32
    memory_size=50000,  # 减少经验回放缓冲区
)
```

### Q3: 如何查看 GPU 使用情况？

**训练时监控 GPU：**

```bash
# 在另一个终端窗口运行
watch -n 1 nvidia-smi
```

或者在 Python 中：
```python
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
```

### Q4: 训练时 GPU 使用率很低？

**可能原因：**
- 批次大小太小
- 网络太小
- 数据预处理是瓶颈

**解决方案：**
- 增加 `batch_size`（64 → 128 或更大）
- 确保数据在 GPU 上（代码已自动处理）

## 📝 完整示例

### 使用 GPU 训练

```python
from agent_Qlearning import DQNAgent, train_dqn_agent
import torch

# 检查 GPU
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU not available, using CPU")

# 创建 agent（自动使用 GPU）
agent = DQNAgent(
    special_pos=None,
    auto_detect_special=True,
    learning_rate=0.001,
    batch_size=128,  # GPU 可以使用更大的批次
    memory_size=200000
)

# 开始训练
train_dqn_agent(
    agent,
    num_episodes=10000,
    save_freq=1000,
    save_path="dqn_2048_model.pth"
)
```

### 使用训练脚本

```bash
# 自动使用 GPU（如果可用）
python train_dqn.py --episodes 10000 --batch-size 128
```

## 🎉 总结

1. **代码已自动支持 GPU** - 无需额外配置
2. **自动检测** - 如果有 GPU 会自动使用
3. **性能提升** - GPU 训练速度可提升 5-20 倍
4. **简单使用** - 直接运行训练脚本即可

**开始训练：**
```bash
python train_dqn.py --episodes 10000
```

代码会自动使用 GPU（如果可用）！🚀



