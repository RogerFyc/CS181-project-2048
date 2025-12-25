# CS181-project-2048 — 2048 with Classic Agents

A simple and brief realization of a modified **2048** game, accompanied with classic agent methods such as **Minimax**, plus brief visualization.

![Game screenshot](img/A_photo_for_the_game.png)

## Features

- 2048 game implementation (Python + Tkinter GUI)
- Classic agent support (e.g., Minimax)
- Easy to modify and extend for experiments

## Installation

Clone this repository:

```bash
git clone https://github.com/RogerFyc/CS181-project-2048.git
cd CS181-project-2048
```

## Getting Started

### Requirements

* Python 3.7 or higher
* Tkinter (usually bundled with Python on many platforms)
* NumPy (for DQN agent)
* PyTorch (optional, only required for DQN agent)

### Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

### Run (GUI)

```bash
python puzzle.py
```

### Train DQN Agent (Optional)

```bash
python train_dqn.py --episodes 500
```

## Project Structure (Typical)

* `puzzle.py` — GUI / entry point
* `logic.py` — core game mechanics (move/merge/spawn/win/lose)
* `constants.py` — constants and UI settings
* `img/` — images used in documentation

## Contributing (Team Workflow)

This repository is mainly for collaboration within our team.
Please use feature branches and open a Pull Request before merging into `main`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Attribution

This repository is a derivative work based on:

* **2048 Python** by Tay Yang Shun (yangshun/2048-python), MIT License

And ultimately inspired by:

* **2048** by Gabriele Cirulli (gabrielecirulli/2048), MIT License

