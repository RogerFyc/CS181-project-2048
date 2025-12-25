# 安装指南

## 📦 必需的Python库

这个项目需要安装以下外部Python库：

### 1. 核心依赖

```bash
pip install numpy
```

### 2. DQN Agent依赖（可选）

如果你想使用DQN Agent，需要安装PyTorch：

**CPU版本（推荐用于快速开始）：**
```bash
pip install torch
```

**GPU版本（如果需要GPU加速训练）：**
```bash
# 访问 https://pytorch.org/get-started/locally/ 查看适合你系统的安装命令
# 例如（CUDA 11.8）:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3. 一键安装（推荐）

使用 `requirements.txt` 文件：

```bash
# 安装所有依赖（CPU版本）
pip install -r requirements.txt
```

---

## ✅ 完整安装步骤

### 方法 1：使用 requirements.txt（推荐）

```bash
# 1. 克隆或下载项目
git clone https://github.com/your-repo/CS181-project-2048.git
cd CS181-project-2048

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行游戏
python puzzle.py
```

### 方法 2：手动安装

```bash
# 1. 安装numpy（必需）
pip install numpy

# 2. 安装PyTorch（如果使用DQN Agent）
pip install torch

# 3. 运行游戏
python puzzle.py
```

---

## 🔍 验证安装

### 检查numpy

```bash
python -c "import numpy; print('numpy version:', numpy.__version__)"
```

### 检查PyTorch（如果安装了）

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### 检查CUDA支持（如果使用GPU）

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 📋 库的用途说明

| 库名 | 用途 | 必需性 |
|------|------|--------|
| **numpy** | DQN Agent的数值计算 | ✅ 必需 |
| **torch** | DQN Agent的深度学习框架 | ⚠️ 仅使用DQN时必需 |
| **tkinter** | GUI界面 | ✅ 必需（Python标准库） |

---

## 🎮 不同功能所需的库

### 基础游戏（Minimax/Expectimax Agent）

只需要：
- Python 3.x
- tkinter（通常随Python安装）
- numpy（项目依赖）

### DQN Agent

额外需要：
- torch (PyTorch)

### 训练DQN Agent

额外需要：
- torch (PyTorch)
- GPU支持（可选，但推荐）

---

## ⚠️ 常见问题

### Q1: 找不到 tkinter 模块？

**Windows/Mac:** tkinter通常随Python一起安装。

**Linux:** 需要安装tkinter：
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch Linux
sudo pacman -S tk
```

### Q2: PyTorch安装失败？

1. **检查Python版本**：需要Python 3.7+
   ```bash
   python --version
   ```

2. **使用官方安装命令**：
   访问 https://pytorch.org/get-started/locally/ 获取适合你系统的安装命令

3. **使用conda（如果pip失败）**：
   ```bash
   conda install pytorch -c pytorch
   ```

### Q3: 提示找不到模块？

确保在项目根目录下运行：
```bash
cd CS181-project-2048
python puzzle.py
```

### Q4: GPU不工作？

1. 检查CUDA是否已安装：
   ```bash
   nvidia-smi
   ```

2. 检查PyTorch是否支持CUDA：
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. 如果返回False，需要安装CUDA版本的PyTorch（见上方GPU安装说明）

---

## 🚀 快速开始

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **运行游戏**：
   ```bash
   python puzzle.py
   ```

3. **训练DQN Agent（可选）**：
   ```bash
   python train_dqn.py --episodes 500
   ```

---

## 📝 版本要求

- **Python**: 3.7 或更高版本
- **numpy**: 1.19.0 或更高版本
- **torch**: 1.9.0 或更高版本（如果使用DQN）

---

## 🔄 更新依赖

```bash
pip install --upgrade -r requirements.txt
```

---

## 📚 更多信息

- [PyTorch安装指南](https://pytorch.org/get-started/locally/)
- [NumPy文档](https://numpy.org/doc/)
- [项目README](README.md)

