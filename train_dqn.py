# train_dqn.py
# DQN Agent 训练脚本
"""
使用此脚本训练 DQN Agent

运行方式： 
    python train_dqn.py

或者修改参数后运行：
    python train_dqn.py --episodes 20000 --save-freq 1000
"""

import argparse
import os
import sys
from agent_Qlearning import DQNAgent, train_dqn_agent
import logic
import constants as c
import random


def main():
    parser = argparse.ArgumentParser(description='Train DQN Agent for 2048')
    
    # ========== 可修改的训练参数 ==========
    # 训练轮数：建议 5000-50000，根据需求调整
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes (default: 5000)')
    
    # 模型保存频率：每N个episode保存一次，建议 100-2000
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Save model every N episodes (default: 100)')
    
    # 模型保存路径：可以修改为其他文件名
    parser.add_argument('--save-path', type=str, default='dqn_2048_model.pth',
                        help='Path to save model (default: dqn_2048_model.pth)')
    
    # 加载已有模型：如果要从已有模型继续训练，设置此参数
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to load existing model (optional)')
    
    # 特殊格位置：如果知道位置可以手动设置，格式: --special-pos 1 1
    parser.add_argument('--special-pos', type=int, nargs=2, default=None,
                        help='Special tile position (i, j). If not provided, auto-detect will be enabled.')
    
    # ========== 可修改的神经网络参数 ==========
    # 学习率：建议 0.0001-0.01，如果训练不稳定可以降低到 0.0005
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='Learning rate (default: 0.001, recommended: 0.0005-0.001)')
    
    # 折扣因子：通常保持 0.95-0.99，不建议修改
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99, usually keep unchanged)')
    
    # ========== 可修改的探索参数 ==========
    # 初始探索率：通常保持 1.0
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Initial epsilon (default: 1.0, usually keep unchanged)')
    
    # 最终探索率：建议 0.01-0.1，如果reward下降可以提高到 0.05
    parser.add_argument('--epsilon-end', type=float, default=0.05,
                        help='Final epsilon (default: 0.05, recommended: 0.01-0.1)')
    
    # 探索率衰减：建议 0.998-0.999，如果reward下降可以改为 0.998
    parser.add_argument('--epsilon-decay', type=float, default=0.998,
                        help='Epsilon decay rate (default: 0.995, recommended: 0.998-0.999)')
    
    # ========== 可修改的经验回放参数 ==========
    # 批次大小：建议 32-128，如果内存充足可以增加到 128
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64, recommended: 32-128)')
    
    # 经验回放缓冲区大小：建议 50000-200000
    parser.add_argument('--memory-size', type=int, default=10000,
                        help='Replay buffer size (default: 100000, recommended: 50000-200000)')
    
    # 目标网络更新频率：建议 500-2000，如果训练不稳定可以降低到 500
    parser.add_argument('--target-update-freq', type=int, default=500,
                        help='Target network update frequency (default: 1000, recommended: 500-2000)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DQN Agent Training for 2048 Game")
    print("=" * 60)
    
    # 确定特殊格位置
    if args.special_pos:
        special_pos = tuple(args.special_pos)
        auto_detect = False
        print(f"Special tile position: {special_pos}")
    else:
        special_pos = None
        auto_detect = True
        print("Special tile position: Auto-detect enabled")
    
    # 创建 agent
    print("\nCreating DQN Agent...")
    agent = DQNAgent(
        special_pos=special_pos,
        auto_detect_special=auto_detect,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq
    )
    
    # 如果提供了加载路径，加载模型
    if args.load_path and os.path.exists(args.load_path):
        print(f"\nLoading existing model from {args.load_path}...")
        try:
            agent.load(args.load_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Starting training from scratch...")
    else:
        print("\nStarting training from scratch...")
    
    # 开始训练
    print(f"\nTraining parameters:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Save frequency: {args.save_freq}")
    print(f"  Save path: {args.save_path}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {agent.device}")
    print("\n" + "=" * 60)
    
    try:
        episode_rewards, episode_lengths = train_dqn_agent(
            agent,
            num_episodes=args.episodes,
            save_freq=args.save_freq,
            save_path=args.save_path
        )
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Final model saved to: {args.save_path}")
        print(f"Average reward (last 100 episodes): {sum(episode_rewards[-100:]) / min(100, len(episode_rewards)):.2f}")
        print(f"Average length (last 100 episodes): {sum(episode_lengths[-100:]) / min(100, len(episode_lengths)):.2f}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Saving current model to {args.save_path}...")
        agent.save(args.save_path)
        print("Model saved.")
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        print(f"Attempting to save current model to {args.save_path}...")
        try:
            agent.save(args.save_path)
            print("Model saved.")
        except:
            print("Failed to save model.")


if __name__ == "__main__":
    main()

