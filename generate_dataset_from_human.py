import json
import os
from datetime import datetime
import numpy as np


def _clone(mat):
    return [row[:] for row in mat]


def flatten_matrix(matrix):
    """将 4x4 矩阵展平为 16 维向量"""
    return np.array(matrix).flatten().tolist()


class DataCollector:
    def __init__(self, output_dir='collected_data', output_format='json'):
        self.output_dir = output_dir
        self.output_format = output_format
        
        self.current_episode_data = []
        self.all_episodes_data = []
        
        self.episode_count = 0
        self.total_steps = 0
        self.total_games = 0
        self.global_episode_id = self._load_global_episode_id()  # ← 唯一的 ID 来源
        
        self.is_recording = False
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"✅ 数据采集器初始化完成")
        print(f"   保存路径: {os.path.abspath(output_dir)}")
        print(f"   下一个 episode ID: {self.global_episode_id}")
    
    def _load_global_episode_id(self):
        """从已保存的文件中读取最大的 episode ID"""
        max_id = 0
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.json') and not filename.endswith('_stats.json'):
                    try:
                        with open(os.path.join(self.output_dir, filename), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for episode in data:
                                    if 'episode' in episode:  
                                        max_id = max(max_id, episode['episode'])
                    except:  
                        pass
        return max_id + 1
    
    def start_recording(self):
        """开始一个新的 episode"""
        if self.current_episode_data:
            self.current_episode_data = []
        
        self.is_recording = True
        self.episode_count += 1
        # ✅ 使用全局编号
        self.current_global_episode_id = self.global_episode_id + self.episode_count - 1
        print(f"▶️  开始记录第 {self.current_global_episode_id} 局（本次运行第 {self.episode_count} 局）")
    
    def record_step(self, state_matrix, action, next_state_matrix, reward, done, special_pos):
        """记录一步"""
        if not self.is_recording:
            return
        
        state = flatten_matrix(state_matrix)
        next_state = flatten_matrix(next_state_matrix)
        
        data_point = {
            'episode': self.current_global_episode_id,  # ✅ 使用全局编号
            'step': len(self.current_episode_data),
            'state': state,
            'action': action,
            'reward': float(reward),
            'special_pos': special_pos,
            'next_state': next_state,
            'done': bool(done)
        }
        
        self.current_episode_data.append(data_point)
        self.total_steps += 1
    
    def stop_recording(self, game_score, game_steps, game_state):
        """结束当前 episode"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.current_episode_data:
            episode_info = {
                'episode': self.current_global_episode_id,  # ✅ 使用全局编号
                'num_steps': len(self.current_episode_data),
                'game_score':  game_score,
                'game_steps': game_steps,
                'game_state': game_state,
                'data': self.current_episode_data
            }
            
            self.all_episodes_data.append(episode_info)
            self.total_games += 1
            
            print(f"✅ 第 {self.current_global_episode_id} 局已保存到内存")
            print(f"   ({len(self.current_episode_data)} 步，分数 {game_score}，{game_state})")
            
            self.current_episode_data = []
    
    def save_to_file(self, filename=None):
        """保存所有数据到文件"""
        if not self.all_episodes_data:
            print("⚠️  没有数据可保存！")
            return False
        
        if filename is None:
            # ✅ 简化：直接用全局编号
            total_games = self._count_total_games_in_files()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"2048_human_data_{total_games + self.total_games}games_{timestamp}"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
                json.dump(self.all_episodes_data, f, indent=2, separators=(', ', ': '), ensure_ascii=False)
            print(f"✅ 数据已保存:  {filepath}. json")
            
            stats = {
                'total_games': self.total_games,
                'total_steps': self.total_steps,
                'timestamp': datetime.now().isoformat(),
                'games':  [
                    {
                        'game_id': ep['episode'],
                        'num_steps': ep['num_steps'],
                        'score': ep['game_score'],
                        'state': ep['game_state']
                    }
                    for ep in self.all_episodes_data
                ]
            }
            
            with open(f"{filepath}_stats.json", 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"📊 统计信息已保存: {filepath}_stats.json")
            
            print(f"\n📈 数据汇总:")
            print(f"   这次保存了 {self.total_games} 局")
            print(f"   这次保存了 {self.total_steps} 步")
            print(f"   平均每局 {self.total_steps / self.total_games:.1f} 步")
            
            self.all_episodes_data = []
            self.total_games = 0
            self.total_steps = 0
            self.episode_count = 0
            # ✅ 更新全局编号
            self.global_episode_id = self._load_global_episode_id()
            print("✅ 已清空本地缓冲，可继续采集\n")
            
            return True
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False
    
    def _count_total_games_in_files(self):
        """统计所有已保存文件中的总局数"""
        total = 0
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.json') and not filename.endswith('_stats.json'):
                    try:
                        with open(os.path.join(self.output_dir, filename), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                total += len(data)
                    except: 
                        pass
        return total
    def discard_current_episode(self):
        """丢弃当前未完成的 episode"""
        if self.is_recording:
            self.is_recording = False
            self.current_episode_data = []