# agent_ActorCritic.py
# Actor-Critic Reinforcement Learning Agent for 2048 with Special Tile
import numpy as np
import random
from pathlib import Path

# Project paths (for default model/data locations)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "Data"
import math
import logic
import constants as c
import os
import json
import glob
import warnings

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
    print("Warning: PyTorch not available. Actor-Critic agent requires PyTorch.")
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
    """计算 log2 空值为 0"""
    return 0 if v <= 0 else int(math.log2(v))


class ActorNetwork(nn.Module):
    """
    Actor (策略网络): 输入状态，输出动作概率分布
    """
    def __init__(self, state_size=18, action_size=4, hidden_size=256):
        """
        state_size: 状态维度 (4x4棋盘 + 2个特殊格坐标 = 16 + 2 = 18)
        action_size: 动作数量 (4个方向)
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # 使用 softmax 输出动作概率分布
        return F.softmax(x, dim=-1)


class CriticNetwork(nn.Module):
    """
    Critic (价值网络): 输入状态，输出状态价值 V(s)
    """
    def __init__(self, state_size=18, hidden_size=256):
        """
        state_size: 状态维度
        """
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)  # 输出单个标量值
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ActorCriticAgent:
    """
    Actor-Critic Reinforcement Learning Agent for 2048 with Special Tile
    """
    
    def __init__(self, 
                 special_pos=None,
                 state_size=18,
                 action_size=4,
                 hidden_size=256,
                 learning_rate_actor=0.001,
                 learning_rate_critic=0.001,
                 gamma=0.99,
                 device=None,
                 auto_detect_special=True):
        """
        初始化 Actor-Critic Agent
        
        Args:
            special_pos: 特殊格子位置 (i, j)
            state_size: 状态维度
            action_size: 动作数量
            hidden_size: 隐藏层大小
            learning_rate_actor: Actor 学习率
            learning_rate_critic: Critic 学习率
            gamma: 折扣因子
            device: 计算设备 (cpu/cuda)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Actor-Critic agent")
        
        self.special_pos = special_pos
        self.auto_detect_special = auto_detect_special
        self.detected_special_pos = None
        self.special_pos_history = []
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Actor 和 Critic 网络
        self.actor = ActorNetwork(state_size, action_size, hidden_size).to(self.device)
        self.critic = CriticNetwork(state_size, hidden_size).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        
        # 奖励函数参数
        self.large_tile_threshold = 32  # 大数字阈值
        self.small_tile_threshold = 8   # 小数字阈值
        self.large_penalty = -100       # 大数字进入特殊格惩罚
        self.small_reward = 10          # 小数字进入特殊格奖励
        self.terminal_penalty = -500    # 游戏结束惩罚
        self.merge_reward_scale = 20    # 合并奖励缩放
        self.step_reward = 1.0          # 每步基础奖励
        self.empty_reward_scale = 2.0   # 空格奖励缩放
        self.max_tile_reward_scale = 5.0  # 最大块奖励缩放
        
        # 特殊格合成计数器
        self.special_merge_count = 0  # 连续在特殊格合成的次数
        self.special_merge_threshold = 5  # 触发强制避免的阈值
    
    def detect_special_position(self, prev_mat, next_mat):
        """
        通过观察移动前后的矩阵变化来检测特殊格位置
        """
        if self.special_pos is not None:
            return self.special_pos
        
        if self.detected_special_pos is not None:
            return self.detected_special_pos
        
        center_start = (4 - 2) // 2
        center_end = center_start + 2
        
        for i in range(center_start, center_end):
            for j in range(center_start, center_end):
                prev_val = prev_mat[i][j]
                next_val = next_mat[i][j]
                
                if prev_val > 2 and next_val == prev_val // 2:
                    self.special_pos_history.append((i, j))
        
        if len(self.special_pos_history) >= 2:
            from collections import Counter
            pos_counts = Counter(self.special_pos_history)
            most_common = pos_counts.most_common(1)
            if most_common and most_common[0][1] >= 2:
                self.detected_special_pos = most_common[0][0]
                return self.detected_special_pos
        
        return None
    
    def update_special_position(self, special_pos):
        """更新特殊格位置"""
        self.special_pos = special_pos
        self.detected_special_pos = special_pos
    
    def get_special_position(self):
        """获取当前使用的特殊格位置"""
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
        """
        state = []
        
        # 棋盘值（log2编码）
        for i in range(4):
            for j in range(4):
                state.append(_log2(mat[i][j]))
        
        # 特殊格位置
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
        
        # 0. 每步基础奖励
        reward += self.step_reward
        
        # 1. 合并奖励
        merge_score = self._calculate_merge_score(state, next_state_mat)
        reward += merge_score * self.merge_reward_scale
        
        # 2. 空格奖励
        next_empty = sum(1 for i in range(4) for j in range(4) if next_state_mat[i][j] == 0)
        reward += next_empty * self.empty_reward_scale
        
        # 3. 最大块奖励
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
        """计算合并得分"""
        state_empty = sum(1 for i in range(4) for j in range(4) if state_mat[i][j] == 0)
        next_empty = sum(1 for i in range(4) for j in range(4) if next_state_mat[i][j] == 0)
        
        merge_count = next_empty - state_empty
        
        if merge_count <= 0:
            return 0.0
        
        state_sum = sum(sum(row) for row in state_mat)
        next_sum = sum(sum(row) for row in next_state_mat)
        
        merge_value = 0.0
        if merge_count > 0:
            value_gain = next_sum - state_sum - 2
            if value_gain > 0:
                merge_value = _log2(value_gain) if value_gain > 0 else 0
            else:
                merge_value = merge_count * 2
        
        return merge_value
    
    def _calculate_special_tile_penalty(self, state_mat, next_state_mat):
        """计算特殊格惩罚/奖励"""
        special_pos = self.get_special_position()
        if special_pos is None:
            return 0.0
        
        i, j = special_pos
        penalty = 0.0
        
        state_value = state_mat[i][j]
        next_value = next_state_mat[i][j]
        
        # 如果特殊格上的值增加了，说明有tile进入了
        if next_value > state_value and next_value > 2:
            if next_value >= self.large_tile_threshold:
                penalty += self.large_penalty
            elif next_value <= self.small_tile_threshold:
                penalty += self.small_reward
        
        return penalty
    
    def sample_action(self, state_mat, training=True):
        """
        从策略分布中采样动作
        
        Args:
            state_mat: 当前状态矩阵
            training: 是否在训练模式
            
        Returns:
            action: 动作索引 (0=Up, 1=Down, 2=Left, 3=Right)
        """
        try:
            state_vector = self.state_to_vector(state_mat)
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_probs = self.actor(state_tensor)
                if training:
                    # 训练时：按概率分布采样
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample().item()
                else:
                    # 测试时：选择概率最大的动作
                    action = action_probs.argmax().item()
            
            return action
        except Exception as e:
            print(f"Error in sample_action: {e}, returning random action")
            return random.randrange(self.action_size)
    
    def update_networks(self, trajectory):
        """
        使用轨迹更新 Actor 和 Critic 网络
        
        Args:
            trajectory: 轨迹列表，每个元素为 (state_vector, action, reward)
        """
        if len(trajectory) == 0:
            return
        
        # 1. 计算回报（Return）
        returns = []
        G = 0
        for _, _, reward in reversed(trajectory):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # 转换为 tensor（先转换为 numpy 数组以提高性能）
        states = torch.FloatTensor(np.array([s for s, _, _ in trajectory])).to(self.device)
        actions = torch.LongTensor(np.array([a for _, a, _ in trajectory])).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        
        # 2. 更新 Critic（价值网络）
        values = self.critic(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 3. 更新 Actor（策略网络）
        # 重新计算 values（因为 critic 已更新）
        with torch.no_grad():
            values = self.critic(states).squeeze()
        
        # 计算 advantage
        advantages = returns - values
        
        # 获取动作概率
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # 策略损失（负对数似然加权 advantage）
        policy_loss = -(log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return value_loss.item(), policy_loss.item()
    
    def _will_trigger_special_merge(self, mat, move_fn):
        """
        检测某个移动是否会导致特殊格上的值被减半（合成）
        
        Args:
            mat: 当前游戏矩阵
            move_fn: 移动函数
            
        Returns:
            bool: 如果会导致特殊格合成返回 True
        """
        special_pos = self.get_special_position()
        if special_pos is None:
            return False
        
        i, j = special_pos
        current_value = mat[i][j]
        
        # 如果特殊格当前为空或值 <= 2，不会触发合成
        if current_value <= 2:
            return False
        
        # 执行移动
        next_mat, done = move_fn(_clone(mat))
        if not done:
            return False
        
        # 应用特殊格效果
        next_mat = _apply_special_cell_effect(next_mat, special_pos)
        next_value = next_mat[i][j]
        
        # 如果值被减半了，说明触发了合成
        if next_value == current_value // 2 and current_value > 2:
            return True
        
        return False
    
    def choose_move(self, mat, prev_mat=None):
        """
        选择移动（用于游戏接口）
        
        Args:
            mat: 当前游戏矩阵
            prev_mat: 上一个状态矩阵（用于检测特殊格位置）
            
        Returns:
            move_name: 移动方向名称 ('Up', 'Down', 'Left', 'Right') 或 None
        """
        # 如果启用了自动检测且还没有检测到特殊格位置，尝试检测
        if self.auto_detect_special and prev_mat is not None:
            self.detect_special_position(prev_mat, mat)
        
        # 检查是否有可用移动
        available_moves = []
        for move_name, move_fn in MOVE_FUNCS.items():
            next_mat, done = move_fn(_clone(mat))
            if done:
                available_moves.append((move_name, move_fn))
        
        if not available_moves:
            return None
        
        # 如果连续5次在特殊格合成，强制选择不会导致特殊格合成的方向
        if self.special_merge_count >= self.special_merge_threshold:
            safe_moves = []
            for move_name, move_fn in available_moves:
                if not self._will_trigger_special_merge(mat, move_fn):
                    safe_moves.append((move_name, move_fn))
            
            if safe_moves:
                # 从安全动作中随机选择一个
                chosen_move = random.choice(safe_moves)
                self.special_merge_count = 0  # 重置计数器
                return chosen_move[0]
            # 如果没有安全动作，继续正常流程（虽然理论上不应该发生）
        
        # 使用 Actor 网络选择动作
        try:
            action = self.sample_action(mat, training=False)
            action_name = MOVE_NAMES[action]
            
            # 检查选择的动作是否会导致特殊格合成
            if action_name in [m[0] for m in available_moves]:
                move_fn = MOVE_FUNCS[action_name]
                if self._will_trigger_special_merge(mat, move_fn):
                    self.special_merge_count += 1
                else:
                    self.special_merge_count = 0  # 重置计数器
                
                return action_name
        except Exception as e:
            print(f"Actor network error: {e}, falling back to random selection")
        
        # 如果选择的动作不可用，或网络出错，选择第一个可用动作
        chosen_move = available_moves[0]
        # 检查是否会导致特殊格合成
        if self._will_trigger_special_merge(mat, chosen_move[1]):
            self.special_merge_count += 1
        else:
            self.special_merge_count = 0
        
        return chosen_move[0]
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hidden_size': self.actor.fc1.out_features,  # 保存 hidden_size
            'state_size': self.state_size,
            'action_size': self.action_size,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        import sys
        import io
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 尝试从 checkpoint 中推断 hidden_size（如果未保存）
        if 'hidden_size' not in checkpoint:
            # 从 actor_state_dict 中推断 hidden_size
            actor_state = checkpoint.get('actor_state_dict', {})
            if 'fc1.weight' in actor_state:
                # fc1.weight 的形状是 [hidden_size, state_size]
                inferred_hidden_size = actor_state['fc1.weight'].shape[0]
                checkpoint['hidden_size'] = inferred_hidden_size
                checkpoint['state_size'] = actor_state['fc1.weight'].shape[1]
                checkpoint['action_size'] = actor_state['fc4.weight'].shape[0] if 'fc4.weight' in actor_state else self.action_size
        
        # 检查是否有保存的 hidden_size
        if 'hidden_size' in checkpoint:
            saved_hidden_size = checkpoint['hidden_size']
            current_hidden_size = self.actor.fc1.out_features
            
            # 如果 hidden_size 不匹配，需要重新创建网络
            if saved_hidden_size != current_hidden_size:
                # 保存当前学习率
                actor_lr = self.actor_optimizer.param_groups[0]['lr']
                critic_lr = self.critic_optimizer.param_groups[0]['lr']
                
                # 重新创建网络
                self.actor = ActorNetwork(
                    state_size=checkpoint.get('state_size', self.state_size),
                    action_size=checkpoint.get('action_size', self.action_size),
                    hidden_size=saved_hidden_size
                ).to(self.device)
                
                self.critic = CriticNetwork(
                    state_size=checkpoint.get('state_size', self.state_size),
                    hidden_size=saved_hidden_size
                ).to(self.device)
                
                # 重新创建优化器
                self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 抑制所有输出（包括警告和错误）
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = io.StringIO()
        sys.stdout = io.StringIO()
        
        try:
            # 加载模型参数（使用 strict=False 允许部分加载）
            try:
                self.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                self.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
            except Exception as e:
                # 即使 strict=False，某些情况下仍可能抛出异常，忽略它
                pass
        finally:
            # 恢复 stdout 和 stderr
            sys.stderr = old_stderr
            sys.stdout = old_stdout
        
        # 加载优化器状态（如果 hidden_size 匹配）
        if 'hidden_size' not in checkpoint or checkpoint['hidden_size'] == self.actor.fc1.out_features:
            try:
                if 'actor_optimizer_state_dict' in checkpoint:
                    self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                if 'critic_optimizer_state_dict' in checkpoint:
                    self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            except Exception:
                # 如果优化器状态不匹配 忽略）
                pass


def train_actor_critic_agent(agent, num_episodes=10000, save_freq=1000, save_path=str(DATA_DIR / "actor_critic_2048_model.pth")):
    """
    训练 Actor-Critic Agent
    
    Args:
        agent: Actor-Critic Agent 实例
        num_episodes: 训练轮数
        save_freq: 保存频率
        save_path: 保存路径
    """
    print("Starting Actor-Critic training...")
    print(f"Device: {agent.device}")
    print(f"Special tile position: {agent.get_special_position()}")
    print(f"Auto-detect special: {agent.auto_detect_special}")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        # 初始化游戏
        state_mat = logic.new_game(c.GRID_LEN)
        
        # 重置特殊格合成计数器（每个episode开始时重置）
        agent.special_merge_count = 0
        
        trajectory = []
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 1. 使用 choose_move 选择动作（包含特殊格合成检测逻辑）
            # 在训练时，我们需要保存上一个状态用于检测
            prev_state_mat = _clone(state_mat) if steps > 0 else None
            
            # 使用 choose_move 方法，它会自动处理特殊格合成检测和强制避免
            action_name = agent.choose_move(state_mat, prev_mat=prev_state_mat)
            
            if action_name is None:
                # 没有可用动作，游戏结束
                done = True
                break
            
            # 2. 执行动作
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
                    if not move_done:
                        done = True
                        break
                else:
                    done = True
                    break
            
            # 获取动作索引（用于记录轨迹）
            action_idx = MOVE_NAMES.index(action_name)
            
            # 应用特殊格效果
            special_pos = agent.get_special_position()
            next_mat = _apply_special_cell_effect(next_mat, special_pos)
            
            # 检测特殊格位置（如果启用自动检测且还没有检测到）
            if agent.auto_detect_special and special_pos is None:
                agent.detect_special_position(state_mat, next_mat)
                special_pos = agent.get_special_position()
            
            # 随机生成新tile
            next_mat = logic.add_two(next_mat)
            
            # 检查游戏状态
            game_state = logic.game_state(next_mat)
            if game_state != 'not over':
                done = True
            
            # 3. 计算奖励
            reward = agent.calculate_reward(state_mat, action_idx, next_mat, done)
            total_reward += reward
            
            # 4. 记录轨迹
            state_vector = agent.state_to_vector(state_mat)
            trajectory.append((state_vector, action_idx, reward))
            
            # 更新状态
            state_mat = next_mat
            steps += 1
        
        # 回合结束后：更新 Actor & Critic
        if len(trajectory) > 0:
            value_loss, policy_loss = agent.update_networks(trajectory)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f}")
        
        # 保存模型
        if (episode + 1) % save_freq == 0:
            agent.save(save_path)
    
    print("Training completed!")
    agent.save(save_path)
    return episode_rewards, episode_lengths


def load_collected_data(data_dir=str(DATA_DIR / "collected_data_try512")):
    """
    从 collected_data 目录加载数据集
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        trajectories: 轨迹列表，每个轨迹包含 (state, action, reward, next_state, done)
    """
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return []
    
    all_trajectories = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for episode_data in data:
                episode_trajectory = []
                steps = episode_data.get('data', [])
                
                for step_data in steps:
                    state = step_data.get('state', [])
                    action = step_data.get('action', '')
                    next_state = step_data.get('next_state', [])
                    
                    # 转换为矩阵格式
                    state_mat = [[state[i*4+j] for j in range(4)] for i in range(4)]
                    next_state_mat = [[next_state[i*4+j] for j in range(4)] for i in range(4)]
                    
                    # 获取特殊格位置
                    special_pos = step_data.get('special_pos', None)
                    if special_pos:
                        special_pos = tuple(special_pos)
                    
                    episode_trajectory.append({
                        'state': state_mat,
                        'action': action,
                        'next_state': next_state_mat,
                        'special_pos': special_pos
                    })
                
                if len(episode_trajectory) > 0:
                    all_trajectories.append(episode_trajectory)
        
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"Loaded {len(all_trajectories)} episodes from {len(json_files)} files")
    return all_trajectories


def train_from_collected_data(agent, data_dir=str(DATA_DIR / "collected_data"), num_epochs=10):
    """
    使用 collected_data 数据集训练 Actor-Critic Agent
    
    Args:
        agent: Actor-Critic Agent 实例
        data_dir: 数据目录路径
        num_epochs: 训练轮数
    """
    print("Loading collected data...")
    trajectories = load_collected_data(data_dir)
    
    if len(trajectories) == 0:
        print("No data found. Please ensure collected_data directory contains JSON files.")
        return
    
    print(f"Training on {len(trajectories)} episodes for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_rewards = []
        
        for episode_idx, episode_trajectory in enumerate(trajectories):
            # 设置特殊格位置（如果数据中有）
            if episode_trajectory[0].get('special_pos'):
                agent.update_special_position(episode_trajectory[0]['special_pos'])
            
            # 构建训练轨迹
            training_trajectory = []
            total_reward = 0
            
            for step_idx, step_data in enumerate(episode_trajectory):
                state_mat = step_data['state']
                action_name = step_data['action']
                next_state_mat = step_data['next_state']
                
                # 转换为动作索引
                if action_name in MOVE_NAMES:
                    action_idx = MOVE_NAMES.index(action_name)
                else:
                    continue
                
                # 检查游戏是否结束
                game_state = logic.game_state(next_state_mat)
                done = (game_state != 'not over')
                
                # 计算奖励
                reward = agent.calculate_reward(state_mat, action_idx, next_state_mat, done)
                total_reward += reward
                
                # 添加到轨迹
                state_vector = agent.state_to_vector(state_mat)
                training_trajectory.append((state_vector, action_idx, reward))
            
            # 更新网络
            if len(training_trajectory) > 0:
                agent.update_networks(training_trajectory)
                epoch_rewards.append(total_reward)
        
        # 打印进度
        if len(epoch_rewards) > 0:
            avg_reward = np.mean(epoch_rewards)
            print(f"Epoch {epoch + 1}/{num_epochs} | Avg Reward: {avg_reward:.2f}")
    
    print("Training from collected data completed!")



if __name__ == "__main__":
    agent = ActorCriticAgent(
        special_pos=None,
        auto_detect_special=True,
        learning_rate_actor=0.001,
        learning_rate_critic=0.001,
        gamma=0.99
    )
    
    print("Actor-Critic Agent initialized.")
    print("Use train_actor_critic_agent() to train from scratch")
    print("Use train_from_collected_data() to train from collected_data")

