# 项目依赖库说明

## 📦 需要安装的外部库

本项目需要安装以下Python库：

### 1. **numpy** ✅ 必需
- **用途**: DQN Agent的数值计算
- **安装**: `pip install numpy`
- **版本要求**: >= 1.19.0

### 2. **torch** (PyTorch) ⚠️ 仅使用DQN时必需
- **用途**: DQN Agent的深度学习框架
- **安装**: 
  - CPU版本: `pip install torch`
  - GPU版本: 访问 https://pytorch.org/get-started/locally/
- **版本要求**: >= 1.9.0

---

## 📚 Python标准库（无需安装）

以下库是Python标准库，通常随Python一起安装，无需额外安装：

| 库名 | 用途 | 文件 |
|------|------|------|
| `tkinter` | GUI界面 | `puzzle.py` |
| `random` | 随机数生成 | 多个文件 |
| `math` | 数学函数 | `agent_Expectimax.py`, `agent_Minimax.py`, `agent_Qlearning.py` |
| `os` | 操作系统接口 | `puzzle.py`, `train_dqn.py`, `agent_Qlearning.py` |
| `collections` | 集合类 | `agent_Qlearning.py` (deque) |
| `pickle` | 对象序列化 | `agent_Qlearning.py` |
| `argparse` | 命令行参数解析 | `train_dqn.py` |
| `sys` | 系统相关参数 | `train_dqn.py` |
| `functools` | 函数工具 | `agent_Minimax.py` (lru_cache) |

---

## 🎯 不同功能所需的库

### 基础游戏 + Minimax/Expectimax Agent

**必需:**
- ✅ Python 3.7+
- ✅ tkinter（Python标准库）
- ✅ numpy

**无需:**
- ❌ PyTorch

### DQN Agent（使用预训练模型）

**必需:**
- ✅ Python 3.7+
- ✅ tkinter（Python标准库）
- ✅ numpy
- ✅ torch (PyTorch)

### 训练DQN Agent

**必需:**
- ✅ Python 3.7+
- ✅ tkinter（Python标准库）
- ✅ numpy
- ✅ torch (PyTorch)

**推荐（可选）:**
- 💡 GPU支持（CUDA）用于加速训练

---

## 🚀 快速安装

### 方法 1：使用 requirements.txt（推荐）

```bash
pip install -r requirements.txt
```

### 方法 2：手动安装

```bash
# 安装numpy（必需）
pip install numpy

# 安装PyTorch（如果使用DQN）
pip install torch
```

---

## ✅ 验证安装

运行以下命令验证库是否已正确安装：

```bash
# 检查numpy
python -c "import numpy; print('✓ numpy', numpy.__version__)"

# 检查PyTorch（如果安装了）
python -c "import torch; print('✓ torch', torch.__version__)"

# 检查CUDA支持（如果使用GPU）
python -c "import torch; print('✓ CUDA:', torch.cuda.is_available())"
```

---

## 📋 完整的导入列表

### puzzle.py
- `tkinter` (标准库)
- `random` (标准库)
- `os` (标准库)

### agent_Qlearning.py
- `numpy` ⚠️ 需要安装
- `torch` ⚠️ 需要安装（如果使用DQN）
- `random` (标准库)
- `math` (标准库)
- `collections` (标准库)
- `pickle` (标准库)
- `os` (标准库)

### agent_Expectimax.py
- `math` (标准库)
- `random` (标准库)

### agent_Minimax.py
- `math` (标准库)
- `random` (标准库)
- `functools` (标准库)

### train_dqn.py
- `argparse` (标准库)
- `os` (标准库)
- `sys` (标准库)
- `random` (标准库)

### logic.py
- `random` (标准库)

---

## 🔧 安装问题排查

### 问题1: ImportError: No module named 'numpy'
**解决方案**: `pip install numpy`

### 问题2: ImportError: No module named 'torch'
**解决方案**: `pip install torch`

### 问题3: ImportError: No module named 'tkinter'
**解决方案**:
- Windows/Mac: 通常随Python安装
- Linux: `sudo apt-get install python3-tk` (Ubuntu/Debian)

### 问题4: PyTorch CUDA不可用
**解决方案**: 安装CUDA版本的PyTorch，见 [INSTALLATION.md](INSTALLATION.md)

---

## 📖 更多信息

- [完整安装指南](INSTALLATION.md)
- [项目README](README.md)
- [PyTorch官网](https://pytorch.org/)
- [NumPy官网](https://numpy.org/)

