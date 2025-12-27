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
                 state_size=18,  # 可修改：状态向量维度（通常不需要修改）
                 action_size=4,  # 可修改：动作数量（4个方向，通常不需要修改）
                 hidden_size=256,  # 可修改：隐藏层大小，建议 128-512，值越大网络容量越大但训练更慢
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
        
        # 学习率调度器（可选，用于在训练后期降低学习率）
        self.learning_rate = learning_rate
        self.scheduler = None  # 可以在训练过程中添加学习率调度器
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(memory_size)
        
        # ========== 可修改的奖励函数参数 ==========
        # 注意：修改这些参数会影响agent的学习行为，建议根据训练效果调整
        
        # 大数字阈值（用于分级惩罚）
        self.large_tile_threshold = 32  # 可修改：建议 16-64
        
        # 数字2进入特殊格奖励（鼓励2进入特殊格）
        self.tile_2_reward = 20  # 可修改：建议 10-50，值越大越鼓励2进入特殊格
        
        # 基础惩罚（>2的数字进入特殊格的基础惩罚）
        self.base_penalty = -200  # 可修改：建议 -100 到 -500，值越小（越负）惩罚越重
        
        # 游戏结束惩罚（游戏失败时的惩罚）
        self.terminal_penalty = -3000  # 可修改：建议 -1000 到 -5000，值越小（越负）惩罚越重
        
        # 合并奖励缩放（合并tile时的奖励倍数）
        self.merge_reward_scale = 20  # 可修改：建议 10-50，值越大越鼓励合并
        
        # 每步基础奖励（鼓励持续游戏，避免过早结束）
        self.step_reward = 1.0  # 可修改：建议 0.5-2.0，值越大越鼓励持续游戏
        
        # 空格奖励缩放（保持空格数量的奖励倍数）
        self.empty_reward_scale = 2.0  # 可修改：建议 1.0-5.0，值越大越鼓励保持空格
        
        # 最大块奖励缩放（创造更大数字的奖励倍数）
        self.max_tile_reward_scale = 5.0  # 可修改：建议 2.0-10.0，值越大越鼓励创造大数字
        
        # 惩罚缩放因子（根据数字大小设置不同惩罚，数字越大惩罚越重）
        self.penalty_scale_factor = 100  # 可修改：建议 50-200，值越大对大数据惩罚越重
        
        # 无效合并惩罚（外部2和特殊格中的2在特殊格合并成4，但除以2变成2的情况）
        self.ineffective_merge_penalty = -150  # 可修改：建议 -100 到 -300，惩罚无效合并
    
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
        
        # 0. 每步基础奖励（鼓励持续游戏）
        reward += self.step_reward
        
        # 1. 合并奖励（基于合并产生的值）
        merge_score = self._calculate_merge_score(state, next_state_mat)
        reward += merge_score * self.merge_reward_scale
        
        # 2. 空格奖励（保持空格很重要）
        next_empty = sum(1 for i in range(4) for j in range(4) if next_state_mat[i][j] == 0)
        reward += next_empty * self.empty_reward_scale
        
        # 3. 最大块奖励（鼓励创造更大的数字）
        max_tile = max(max(row) for row in next_state_mat)
        if max_tile > 0:
            max_tile_log = _log2(max_tile)
            reward += max_tile_log * self.max_tile_reward_scale
        
        # 4. 特殊格惩罚/奖励
        special_penalty = self._calculate_special_tile_penalty(state, next_state_mat)
        reward += special_penalty
        
        # 5. 游戏结束惩罚/奖励
        if done:
            if logic.game_state(next_state_mat) == 'lose':
                reward += self.terminal_penalty
            elif logic.game_state(next_state_mat) == 'win':
                reward += 1000  # 胜利奖励
        
        return reward
    
    def _calculate_merge_score(self, state_mat, next_state_mat):
        """
        计算合并得分（基于合并产生的值，而不仅仅是数量）
        返回合并产生的总价值（log2编码）
        """
        # 检测空格数量变化（合并会产生空格）
        state_empty = sum(1 for i in range(4) for j in range(4) if state_mat[i][j] == 0)
        next_empty = sum(1 for i in range(4) for j in range(4) if next_state_mat[i][j] == 0)
        
        # 空格增加 = 合并发生
        merge_count = next_empty - state_empty
        
        if merge_count <= 0:
            return 0.0
        
        # 计算合并产生的总价值
        # 方法：比较移动前后的总和，合并会增加总和（因为两个相同数字合并成一个翻倍的数字）
        state_sum = sum(sum(row) for row in state_mat)
        next_sum = sum(sum(row) for row in next_state_mat)
        
        # 合并产生的额外价值 = 新总和 - 旧总和
        # 但还需要考虑新生成的tile（通常是2）
        # 简化处理：如果空格增加，说明有合并，给予基于合并数量的奖励
        # 更准确的方法：检测实际合并的值
        merge_value = 0.0
        
        # 尝试检测合并的位置和值
        # 由于合并逻辑复杂，我们使用简化方法：基于空格增加和总和变化
        if merge_count > 0:
            # 估算合并产生的价值（基于总和变化）
            value_gain = next_sum - state_sum - 2  # 减去新生成的tile（通常是2）
            if value_gain > 0:
                # 合并产生的价值（log2编码）
                merge_value = _log2(value_gain) if value_gain > 0 else 0
            else:
                # 如果无法准确计算，使用合并数量作为基础
                merge_value = merge_count * 2  # 每个合并至少产生2的价值
        
        return merge_value
    
    def _calculate_special_tile_penalty(self, state_mat, next_state_mat):
        """
        计算特殊格惩罚/奖励
        策略：只有数字2可以进入特殊格（给予奖励），其他所有数字（>2）进入都受到惩罚
        根据进入特殊格的数字大小设置分级惩罚（数字越大惩罚越重）
        同时检测无效合并：外部2和特殊格中的2在特殊格合并成4，但除以2变成2的情况
        """
        special_pos = self.get_special_position()
        if special_pos is None:
            return 0.0
        
        i, j = special_pos
        penalty = 0.0
        
        # 检查是否有tile进入了特殊格
        state_value = state_mat[i][j]
        next_value = next_state_mat[i][j]
        
        # ========== 检测无效合并：外部2和特殊格中的2在特殊格合并成4，但除以2变成2 ==========
        # 这种情况的特征：
        # 1. 移动前特殊格是2
        # 2. 移动后特殊格仍然是2（因为4被特殊格效果减半成2）
        # 3. 发生了合并（两个2合并成4，然后4被特殊格效果减半成2）
        # 4. 这实际上是一个无效的合并，因为最终结果还是2，没有实际收益
        
        if state_value == 2 and next_value == 2:
            # 检查是否发生了合并
            # 方法：检查空格数量变化（合并会产生空格）
            state_empty = sum(1 for row in state_mat for cell in row if cell == 0)
            next_empty = sum(1 for row in next_state_mat for cell in row if cell == 0)
            
            # 如果空格增加，说明发生了合并
            # 如果特殊格前后都是2，但空格增加了，很可能是无效合并
            if next_empty > state_empty:
                # 检查是否有2被推入特殊格并发生无效合并
                # 方法：统计移动前后2的数量变化
                # 无效合并的特征：
                # - 两个2合并成4，然后4被特殊格效果减半成2
                # - 所以2的总数应该减少1（两个2变成一个2）
                # - 但新生成的tile通常是2，所以2的总数可能不变
                # - 更准确的检测：检查特殊格所在的行和列是否有2消失
                
                # 检查特殊格所在的行和列，看是否有2消失（被推入特殊格）
                has_2_merged_in_special = False
                
                # 检查特殊格所在的行
                row_2_before = sum(1 for col in range(4) if state_mat[i][col] == 2)
                row_2_after = sum(1 for col in range(4) if next_state_mat[i][col] == 2)
                if row_2_before > row_2_after:
                    has_2_merged_in_special = True
                
                # 检查特殊格所在的列
                if not has_2_merged_in_special:
                    col_2_before = sum(1 for row in range(4) if state_mat[row][j] == 2)
                    col_2_after = sum(1 for row in range(4) if next_state_mat[row][j] == 2)
                    if col_2_before > col_2_after:
                        has_2_merged_in_special = True
                
                # 如果检测到无效合并，给予惩罚
                if has_2_merged_in_special:
                    penalty += self.ineffective_merge_penalty
        
        # ========== 原有逻辑：检查是否有tile进入了特殊格 ==========
        # 如果特殊格上的值增加了，说明有tile进入了
        if next_value > state_value:
            # 只有数字2可以进入特殊格，给予奖励
            if next_value == 2:
                penalty += self.tile_2_reward
            # 所有大于2的数字进入特殊格都受到惩罚
            elif next_value > 2:
                # 基础惩罚（所有>2的数字都有）
                penalty += self.base_penalty
                
                # 根据数字大小（log2编码）增加分级惩罚
                # 例如：4 -> log2=2, 8 -> log2=3, 16 -> log2=4, 32 -> log2=5, 64 -> log2=6, 128 -> log2=7
                tile_log = _log2(next_value)
                # 从4开始计算（log2(4)=2），每增加1级（翻倍），惩罚增加
                base_log = 2  # log2(4) = 2
                extra_penalty = (tile_log - base_log) * self.penalty_scale_factor
                penalty += extra_penalty
                
                # 特别严重的惩罚：如果数字 >= 16
                if next_value >= 16:
                    penalty += -300  # 额外严重惩罚
                # 如果数字 >= 32
                if next_value >= 32:
                    penalty += -500  # 更严重惩罚
                # 如果数字 >= 64（接近获胜条件128的一半）
                if next_value >= 64:
                    penalty += -1000  # 极严重惩罚
                # 如果数字 >= 128（达到或超过获胜条件）
                if next_value >= 128:
                    penalty += -2000  # 最严重惩罚（不应该让获胜数字进入特殊格）
        
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
        # 使用软更新（可选，更稳定）或硬更新（当前方式）
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            # 硬更新：直接复制权重
            self.target_network.load_state_dict(self.q_network.state_dict())
            # 可选：软更新（更稳定，但需要额外的tau参数）
            # tau = 0.01  # 软更新系数
            # for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            #     target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
        # 注意：epsilon衰减应该在每个episode结束时进行，而不是每次replay时
        # 这样可以避免epsilon衰减过快
        
        return loss.item()
    
    def decay_epsilon(self, episode=None, total_episodes=None):
        """
        在每个episode结束时衰减探索率
        
        Args:
            episode: 当前episode编号（可选，用于线性衰减）
            total_episodes: 总episode数（可选，用于线性衰减）
        """
        if self.epsilon > self.epsilon_end:
            # 使用指数衰减（默认）
            self.epsilon *= self.epsilon_decay
            # 确保epsilon不会低于最小值
            self.epsilon = max(self.epsilon, self.epsilon_end)
    
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
    
    def _would_cause_dead_loop(self, mat, move_name):
        """
        检测某个移动是否会导致死循环。
        死循环情况：只有1-2个空格子，在特殊格附近，特殊格中是2，
        如果向特殊格方向移动，会将新合成的4推入特殊格变成2，形成死循环。
        
        Args:
            mat: 当前游戏矩阵
            move_name: 移动方向名称
            
        Returns:
            bool: 如果会导致死循环返回True，否则返回False
        """
        special_pos = self.get_special_position()
        if special_pos is None:
            return False
        
        # 检查空格子数量（1-2个）
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if mat[i][j] == 0:
                    empty_cells.append((i, j))
        
        if len(empty_cells) < 1 or len(empty_cells) > 2:
            return False
        
        special_i, special_j = special_pos
        
        # 检查特殊格中是否是2
        if mat[special_i][special_j] != 2:
            return False
        
        # 检查空格子是否在特殊格附近（上下左右，或对角线相邻）
        # 对于2个空格子的情况，至少有一个应该在特殊格旁边
        has_adjacent_empty = False
        for empty_pos in empty_cells:
            is_adjacent = (abs(empty_pos[0] - special_i) == 1 and empty_pos[1] == special_j) or \
                          (abs(empty_pos[1] - special_j) == 1 and empty_pos[0] == special_i)
            if is_adjacent:
                has_adjacent_empty = True
                break
        
        if not has_adjacent_empty:
            return False
        
        # 检查移动方向是否会向特殊格方向移动
        # 获取移动函数
        move_fn = MOVE_FUNCS.get(move_name)
        if move_fn is None:
            return False
        
        # 模拟移动
        next_mat, done = move_fn(_clone(mat))
        if not done:
            return False
        
        # 应用特殊格效果
        next_mat = _apply_special_cell_effect(next_mat, special_pos)
        
        # 检查移动后特殊格的值
        next_special_val = next_mat[special_i][special_j]
        next_empty_cells = [(i, j) for i in range(4) for j in range(4) if next_mat[i][j] == 0]
        
        # 检查移动后是否会产生死循环状态
        # 死循环的特征：
        # 1. 移动后特殊格变成2（说明有4被推入并减半）
        # 2. 移动后空格子还在（说明没有完全填满）
        # 3. 关键：检查移动前在移动方向上是否有可以合并成4的情况
        
        special_val_before = mat[special_i][special_j]
        
        # 如果移动后特殊格变成2，且空格子还在，需要进一步检查
        # 这可能是死循环的征兆：4被推入特殊格并减半成2，空格子继续变成2
        if next_special_val == 2 and len(next_empty_cells) >= 1:
            # 检查移动方向上是否有两个2可以合并成4
            # 这会合并成4，然后4被推入特殊格变成2
            has_mergeable_2s = False
            
            if move_name == "Down":
                # 检查特殊格所在的列中是否有两个2可以合并
                col = special_j
                for row in range(3):
                    if mat[row][col] == 2 and mat[row+1][col] == 2:
                        has_mergeable_2s = True
                        break
            elif move_name == "Up":
                # 检查特殊格所在的列中是否有两个2可以合并
                col = special_j
                for row in range(1, 4):
                    if mat[row][col] == 2 and mat[row-1][col] == 2:
                        has_mergeable_2s = True
                        break
            elif move_name == "Right":
                # 检查特殊格所在的行中是否有两个2可以合并
                row = special_i
                for col in range(3):
                    if mat[row][col] == 2 and mat[row][col+1] == 2:
                        has_mergeable_2s = True
                        break
            elif move_name == "Left":
                # 检查特殊格所在的行中是否有两个2可以合并
                row = special_i
                for col in range(1, 4):
                    if mat[row][col] == 2 and mat[row][col-1] == 2:
                        has_mergeable_2s = True
                        break
            
            # 如果移动方向上有可合并的2，且移动后特殊格变成2，可能形成死循环
            # 因为两个2会合并成4，然后4被推入特殊格并减半成2
            if has_mergeable_2s:
                return True
            # 如果特殊格之前不是2，但移动后变成2，说明有4（或更大的数字）被推入并减半
            # 这种情况下，如果空格子还在，也可能形成死循环
            # 但为了更准确，我们只在特殊格之前是2的情况下才认为可能是死循环
            # 或者特殊格之前是4，移动后变成2（说明4被推入并减半）
            elif special_val_before == 2 or special_val_before == 4:
                return True
        
        return False
    
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
        
        # 检测并排除会导致死循环的移动
        safe_moves = []
        dead_loop_moves = []
        for move_name, move_fn in available_moves:
            if self._would_cause_dead_loop(mat, move_name):
                dead_loop_moves.append((move_name, move_fn))
            else:
                safe_moves.append((move_name, move_fn))
        
        # 如果有安全的移动，优先使用安全的移动
        if safe_moves:
            available_moves = safe_moves
        # 如果没有安全的移动（理论上不应该发生），使用所有可用移动
        # 但记录警告
        elif dead_loop_moves:
            print("Warning: All available moves would cause dead loop, using them anyway")
            available_moves = dead_loop_moves
        
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
        
        # 在每个episode结束时衰减epsilon（而不是每次replay时）
        agent.decay_epsilon()
        
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
    
    print("Training completed!")
    agent.save(save_path)
    return episode_rewards, episode_lengths


# 使用示例
if __name__ == "__main__":
    agent = DQNAgent(
        special_pos=None,  
        auto_detect_special=True,  
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
