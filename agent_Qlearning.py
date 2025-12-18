# agent_Qlearning.py
# Deep Q-Learning Agent for 2048 with Special Tile
import numpy as np
import random
import math
import logic
import constants as c
from collections import deque
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DQN agent requires PyTorch.")
    print("Install with: pip install torch")


MOVE_FUNCS = {
    "Up": logic.up,
    "Down": logic.down,
    "Left": logic.left,
    "Right": logic.right,
}

MOVE_NAMES = ["Up", "Down", "Left", "Right"]
NUM_ACTIONS = 4


def _clone(mat):
    return [row[:] for row in mat]


def _apply_special_cell_effect(mat, special_pos):
    """改版规则：成功移动后，特殊格子上的值如果 >2，则整除2。"""
    if special_pos is None:
        return mat
    i, j = special_pos
    if mat[i][j] > 2:
        mat[i][j] //= 2
    return mat


def _log2(v):
    """计算 log2，空值为 0"""
    return 0 if v <= 0 else int(math.log2(v))


class QNetwork(nn.Module):
    """
    Deep Q-Network: 输入状态，输出每个动作的Q值
    """
    def __init__(self, state_size=18, action_size=4, hidden_size=256):
        """
        state_size: 状态维度 (4x4棋盘 + 2个特殊格坐标 = 16 + 2 = 18)
        action_size: 动作数量 (4个方向)
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Learning Agent for 2048 with Special Tile
    """
    
    def __init__(self, 
                 special_pos=None,
                 state_size=18,
                 action_size=4,
                 hidden_size=256,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 memory_size=100000,
                 batch_size=64,
                 target_update_freq=1000,
                 device=None,
                 auto_detect_special=True):
        """
        初始化 DQN Agent
        
        Args:
            special_pos: 特殊格子位置 (i, j)
            state_size: 状态维度
            action_size: 动作数量
            hidden_size: 隐藏层大小
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减
            memory_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
            device: 计算设备 (cpu/cuda)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN agent")
        
        self.special_pos = special_pos
        self.auto_detect_special = auto_detect_special
        self.detected_special_pos = None  # 检测到的特殊格位置
        self.special_pos_history = []  # 用于检测特殊格位置的历史记录
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Q-network 和目标网络
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(memory_size)
        
        # 奖励函数参数
        self.large_tile_threshold = 32  # 大数字阈值
        self.small_tile_threshold = 8   # 小数字阈值
        self.large_penalty = -100       # 大数字进入特殊格惩罚
        self.small_reward = 10          # 小数字进入特殊格奖励
        self.terminal_penalty = -1000   # 游戏结束惩罚
        self.merge_reward_scale = 10    # 合并奖励缩放
    
    def detect_special_position(self, prev_mat, next_mat):
        """
        通过观察移动前后的矩阵变化来检测特殊格位置
        特殊格的特征：移动后，如果某个格子的值被减半，且该位置在中心区域，可能是特殊格
        
        Args:
            prev_mat: 移动前的矩阵
            next_mat: 移动后的矩阵（已应用特殊格效果）
            
        Returns:
            detected_pos: 检测到的特殊格位置 (i, j) 或 None
        """
        # 如果已经知道特殊格位置，直接返回
        if self.special_pos is not None:
            return self.special_pos
        
        # 如果已经检测到，直接返回
        if self.detected_special_pos is not None:
            return self.detected_special_pos
        
        # 检测方法：观察哪些格子的值在移动后被减半
        # 特殊格位置通常在中心区域 (1,1) 到 (2,2)
        center_start = (4 - 2) // 2  # = 1
        center_end = center_start + 2  # = 3
        
        for i in range(center_start, center_end):
            for j in range(center_start, center_end):
                prev_val = prev_mat[i][j]
                next_val = next_mat[i][j]
                
                # 如果值被减半（且不是0），可能是特殊格
                if prev_val > 2 and next_val == prev_val // 2:
                    # 记录这个可能的位置
                    self.special_pos_history.append((i, j))
        
        # 如果多次观察到同一个位置被减半，确认它是特殊格
        if len(self.special_pos_history) >= 2:
            # 统计最常出现的位置
            from collections import Counter
            pos_counts = Counter(self.special_pos_history)
            most_common = pos_counts.most_common(1)
            if most_common and most_common[0][1] >= 2:
                self.detected_special_pos = most_common[0][0]
                print(f"Detected special tile position: {self.detected_special_pos}")
                return self.detected_special_pos
        
        return None
    
    def update_special_position(self, special_pos):
        """
        更新特殊格位置（从外部获取）
        
        Args:
            special_pos: 特殊格位置 (i, j)
        """
        self.special_pos = special_pos
        self.detected_special_pos = special_pos
    
    def get_special_position(self):
        """
        获取当前使用的特殊格位置
        
        Returns:
            special_pos: 特殊格位置 (i, j) 或 None
        """
        if self.special_pos is not None:
            return self.special_pos
        return self.detected_special_pos
    
    def state_to_vector(self, mat):
        """
        将游戏状态转换为向量表示
        
        Args:
            mat: 4x4 游戏矩阵
            
        Returns:
            state_vector: 状态向量 (18维)
                - 前16维: 4x4棋盘的值（log2编码，空值为0）
                - 后2维: 特殊格位置 (i, j)
        """
        state = []
        
        # 棋盘值（log2编码）
        for i in range(4):
            for j in range(4):
                state.append(_log2(mat[i][j]))
        
        # 特殊格位置（使用检测到的或已知的）
        special_pos = self.get_special_position()
        if special_pos is not None:
            state.append(special_pos[0] / 4.0)  # 归一化到 [0, 1]
            state.append(special_pos[1] / 4.0)
        else:
            state.append(0.0)
            state.append(0.0)
        
        return np.array(state, dtype=np.float32)
    
    def calculate_reward(self, state, action, next_state_mat, done):
        """
        计算奖励函数
        
        Args:
            state: 当前状态矩阵
            action: 执行的动作
            next_state_mat: 下一个状态矩阵
            done: 游戏是否结束
            
        Returns:
            reward: 奖励值
        """
        reward = 0.0
        
        # 1. 合并奖励
        merge_score = self._calculate_merge_score(state, next_state_mat)
        reward += merge_score * self.merge_reward_scale
        
        # 2. 特殊格惩罚/奖励
        special_penalty = self._calculate_special_tile_penalty(state, next_state_mat)
        reward += special_penalty
        
        # 3. 游戏结束惩罚
        if done:
            if logic.game_state(next_state_mat) == 'lose':
                reward += self.terminal_penalty
            elif logic.game_state(next_state_mat) == 'win':
                reward += 1000  # 胜利奖励
        
        return reward
    
    def _calculate_merge_score(self, state_mat, next_state_mat):
        """计算合并得分"""
        # 计算合并前后的总和差异
        state_sum = sum(sum(row) for row in state_mat)
        next_sum = sum(sum(row) for row in next_state_mat)
        
        # 合并会增加总和（因为两个相同数字合并成一个翻倍的数字）
        # 但实际上，合并后总和不变，我们需要通过其他方式检测
        
        # 更准确的方法：检测空格数量变化
        state_empty = sum(1 for i in range(4) for j in range(4) if state_mat[i][j] == 0)
        next_empty = sum(1 for i in range(4) for j in range(4) if next_state_mat[i][j] == 0)
        
        # 空格增加 = 合并发生
        merge_count = next_empty - state_empty
        
        return merge_count
    
    def _calculate_special_tile_penalty(self, state_mat, next_state_mat):
        """计算特殊格惩罚/奖励"""
        special_pos = self.get_special_position()
        if special_pos is None:
            return 0.0
        
        i, j = special_pos
        penalty = 0.0
        
        # 检查是否有tile进入了特殊格
        state_value = state_mat[i][j]
        next_value = next_state_mat[i][j]
        
        # 如果特殊格上的值增加了，说明有tile进入了
        if next_value > state_value and next_value > 2:
            # 检查进入前的值（可能是移动前的值，或者合并后的值）
            # 简化处理：如果当前值 >= 阈值，给予惩罚
            if next_value >= self.large_tile_threshold:
                penalty += self.large_penalty
            elif next_value <= self.small_tile_threshold:
                penalty += self.small_reward
        
        return penalty
    
    def select_action(self, state_mat, training=True):
        """
        选择动作（epsilon-greedy策略）
        
        Args:
            state_mat: 当前状态矩阵
            training: 是否在训练模式（影响探索率）
            
        Returns:
            action: 动作索引 (0=Up, 1=Down, 2=Left, 3=Right)
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择
            return random.randrange(self.action_size)
        
        # 利用：选择Q值最大的动作
        try:
            state_vector = self.state_to_vector(state_mat)
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
            
            return action
        except Exception as e:
            # 如果Q-network出错，返回随机动作
            print(f"Error in select_action: {e}, returning random action")
            return random.randrange(self.action_size)
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """
        从经验回放缓冲区中学习
        """
        if len(self.memory) < self.batch_size:
            return
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def _is_board_full(self, mat):
        """检查棋盘是否已填满（没有空格）。"""
        n = len(mat)
        for i in range(n):
            for j in range(n):
                if mat[i][j] == 0:
                    return False
        return True
    
    def _has_merge_potential(self, mat, move_fn):
        """
        检查某个移动方向是否可能产生合并。
        方法：检查移动后是否有空格产生（合并会产生空格）。
        """
        next_mat, done = move_fn(_clone(mat))
        if not done:
            return False
        
        # 如果移动后产生了空格，说明有合并
        n = len(mat)
        original_empty = sum(1 for i in range(n) for j in range(n) if mat[i][j] == 0)
        next_empty = sum(1 for i in range(n) for j in range(n) if next_mat[i][j] == 0)
        
        return next_empty > original_empty
    
    def choose_move(self, mat, prev_mat=None):
        """
        选择移动（用于游戏接口）
        优化：当棋盘填满时，优先考虑可以产生合并的方向，避免卡住。
        
        Args:
            mat: 当前游戏矩阵
            prev_mat: 上一个状态矩阵（用于检测特殊格位置）
            
        Returns:
            move_name: 移动方向名称 ('Up', 'Down', 'Left', 'Right') 或 None
        """
        # 如果启用了自动检测且还没有检测到特殊格位置，尝试检测
        if self.auto_detect_special and prev_mat is not None:
            # 尝试检测特殊格位置
            self.detect_special_position(prev_mat, mat)
        
        # 检查是否有可用移动
        available_moves = []
        for move_name, move_fn in MOVE_FUNCS.items():
            next_mat, done = move_fn(_clone(mat))
            if done:
                available_moves.append((move_name, move_fn))
        
        if not available_moves:
            return None
        
        # 检查棋盘是否填满
        is_full = self._is_board_full(mat)
        
        # 如果棋盘填满，优先考虑可以产生合并的方向（避免卡住）
        if is_full:
            moves_with_merges = []
            for move_name, move_fn in available_moves:
                if self._has_merge_potential(mat, move_fn):
                    moves_with_merges.append(move_name)
            
            # 如果有可以合并的方向，优先选择这些方向
            if moves_with_merges:
                # 使用Q-network评估这些可合并的方向，选择最优的
                best_move = None
                best_score = -float("inf")
                
                try:
                    for move_name in moves_with_merges:
                        # 获取移动后的状态
                        move_fn = MOVE_FUNCS[move_name]
                        next_mat, _ = move_fn(_clone(mat))
                        next_mat = _apply_special_cell_effect(next_mat, self.get_special_position())
                        
                        # 使用Q-network评估
                        state_vector = self.state_to_vector(next_mat)
                        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            q_values = self.q_network(state_tensor)
                            # 使用最大Q值作为评分
                            score = q_values.max().item()
                        
                        if score > best_score:
                            best_score = score
                            best_move = move_name
                    
                    if best_move is not None:
                        return best_move
                except Exception as e:
                    # 如果Q-network评估出错，直接返回第一个可合并的方向
                    print(f"Error evaluating moves with Q-network: {e}, using first merge move")
                
                # 如果评估失败或出错，返回第一个可合并的方向
                return moves_with_merges[0]
            # 如果没有可合并的方向，继续正常流程（虽然理论上不应该发生）
        
        # 正常情况：使用Q-network选择动作
        try:
            action = self.select_action(mat, training=False)
            action_name = MOVE_NAMES[action]
            
            # 如果选择的动作可用，直接返回
            if action_name in [m[0] for m in available_moves]:
                return action_name
        except Exception as e:
            # 如果Q-network出错，回退到随机选择可用动作
            print(f"Q-network error: {e}, falling back to random selection")
        
        # 如果选择的动作不可用，或Q-network出错，选择第一个可用动作
        return available_moves[0][0]
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.update_counter = checkpoint.get('update_counter', 0)
        print(f"Model loaded from {filepath}")


def train_dqn_agent(agent, num_episodes=10000, save_freq=1000, save_path="dqn_model.pth"):
    """
    训练 DQN Agent
    
    Args:
        agent: DQN Agent 实例
        num_episodes: 训练轮数
        save_freq: 保存频率
        save_path: 保存路径
    """
    print("Starting DQN training...")
    print(f"Device: {agent.device}")
    print(f"Special tile position: {agent.get_special_position()}")
    print(f"Auto-detect special: {agent.auto_detect_special}")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        # 初始化游戏
        state_mat = logic.new_game(c.GRID_LEN)
        state_vector = agent.state_to_vector(state_mat)
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 选择动作
            action_idx = agent.select_action(state_mat, training=True)
            action_name = MOVE_NAMES[action_idx]
            
            # 执行动作
            move_fn = MOVE_FUNCS[action_name]
            next_mat, move_done = move_fn(_clone(state_mat))
            
            if not move_done:
                # 如果无法移动，尝试其他动作
                available_actions = []
                for name, fn in MOVE_FUNCS.items():
                    test_mat, test_done = fn(_clone(state_mat))
                    if test_done:
                        available_actions.append((name, MOVE_NAMES.index(name)))
                
                if available_actions:
                    action_name, action_idx = random.choice(available_actions)
                    move_fn = MOVE_FUNCS[action_name]
                    next_mat, move_done = move_fn(_clone(state_mat))
                else:
                    done = True
                    break
            
            # 应用特殊格效果（使用检测到的或已知的特殊格位置）
            special_pos = agent.get_special_position()
            next_mat = _apply_special_cell_effect(next_mat, special_pos)
            
            # 检测特殊格位置（如果启用自动检测且还没有检测到）
            if agent.auto_detect_special and special_pos is None:
                agent.detect_special_position(state_mat, next_mat)
                # 重新获取（可能已经检测到）
                special_pos = agent.get_special_position()
            
            # 随机生成新tile
            next_mat = logic.add_two(next_mat)
            
            # 检查游戏状态
            game_state = logic.game_state(next_mat)
            if game_state != 'not over':
                done = True
            
            # 计算奖励
            reward = agent.calculate_reward(state_mat, action_idx, next_mat, done)
            total_reward += reward
            
            # 存储经验
            next_state_vector = agent.state_to_vector(next_mat)
            agent.remember(state_vector, action_idx, reward, next_state_vector, done)
            
            # 学习
            loss = agent.replay()
            
            # 更新状态
            state_mat = next_mat
            state_vector = next_state_vector
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # 保存模型
        if (episode + 1) % save_freq == 0:
            agent.save(save_path)
            print(f"  💾 Model saved at episode {episode + 1}")
    
    print("Training completed!")
    agent.save(save_path)
    return episode_rewards, episode_lengths


# 使用示例
if __name__ == "__main__":
    # 创建agent（两种方式）
    
    # 方式1：提供特殊格位置
    # special_pos = (1, 1)  # 特殊格在中心
    # agent = DQNAgent(
    #     special_pos=special_pos,
    #     auto_detect_special=False,  # 禁用自动检测
    #     learning_rate=0.001,
    #     gamma=0.99,
    #     epsilon_start=1.0,
    #     epsilon_end=0.01,
    #     epsilon_decay=0.995,
    #     memory_size=100000,
    #     batch_size=64,
    #     target_update_freq=1000
    # )
    
    # 方式2：启用自动检测（推荐）
    agent = DQNAgent(
        special_pos=None,  # 不提供特殊格位置
        auto_detect_special=True,  # 启用自动检测
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=100000,
        batch_size=64,
        target_update_freq=1000
    )
    
    # 训练
    # train_dqn_agent(agent, num_episodes=10000, save_path="dqn_2048_model.pth")
    
    # 或者加载已训练的模型
    # agent.load("dqn_2048_model.pth")
    
    print("DQN Agent initialized. Use train_dqn_agent() to train or load() to load a model.")
    print("Auto-detection enabled: agent will detect special tile position during gameplay.")
