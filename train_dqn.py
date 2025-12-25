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
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes (default: 10000)')
    parser.add_argument('--save-freq', type=int, default= 50 ,
                        help='Save model every N episodes (default: 100)')
    parser.add_argument('--save-path', type=str, default='dqn_2048_model.pth',
                        help='Path to save model (default: dqn_2048_model.pth)')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to load existing model (optional)')
    parser.add_argument('--special-pos', type=int, nargs=2, default=None,
                        help='Special tile position (i, j). If not provided, auto-detect will be enabled.')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Initial epsilon (default: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Final epsilon (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate (default: 0.995)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--memory-size', type=int, default=100000,
                        help='Replay buffer size (default: 100000)')
    parser.add_argument('--target-update-freq', type=int, default=1000,
                        help='Target network update frequency (default: 1000)')
    
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

