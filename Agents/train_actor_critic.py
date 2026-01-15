# train_actor_critic.py
# Actor-Critic Agent 训练脚本
"""
使用此脚本训练 Actor-Critic Agent

运行方式：
    python Agents/train_actor_critic.py

或（推荐，作为模块运行）：
    python -m Agents.train_actor_critic

或者修改参数后运行：
    python Agents/train_actor_critic.py --episodes 20000 --save-freq 1000 --use-collected-data
"""

import argparse
import os
import sys
from pathlib import Path

# --- Path bootstrap (allow running this script from inside Agents/) ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DATA_DIR = ROOT_DIR / 'Data'
os.makedirs(DATA_DIR, exist_ok=True)

from Agents.agent_ActorCritic import ActorCriticAgent, train_actor_critic_agent, train_from_collected_data

import logic
import constants as c
import random


def main():
    parser = argparse.ArgumentParser(description='Train Actor-Critic Agent for 2048')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes (default: 500)')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Save model every N episodes (default: 50)')
    parser.add_argument('--save-path', type=str, default=str((Path(__file__).resolve().parent / 'Data' / 'actor_critic_2048_model.pth')),
                        help='Path to save model (default: Data/actor_critic_2048_model.pth)')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to load existing model (optional)')
    parser.add_argument('--special-pos', type=int, nargs=2, default=None,
                        help='Special tile position (i, j). If not provided, auto-detect will be enabled.')
    parser.add_argument('--learning-rate-actor', type=float, default=0.001,
                        help='Actor learning rate (default: 0.001)')
    parser.add_argument('--learning-rate-critic', type=float, default=0.001,
                        help='Critic learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden layer size (default: 256)')
    parser.add_argument('--use-collected-data', action='store_true',
                        help='Train using collected_data directory instead of self-play')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR / 'collected_data_try512'),
                        help='Directory containing collected data (default: Data/collected_data_try512)')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of epochs when training from collected data (default: 10)')
    
    args = parser.parse_args()

    # Normalize paths (treat relative paths as relative to project root)
    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = ROOT_DIR / save_path
    args.save_path = str(save_path)
    if args.load_path:
        load_path = Path(args.load_path)
        if not load_path.is_absolute():
            load_path = ROOT_DIR / load_path
        args.load_path = str(load_path)
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = ROOT_DIR / data_dir
        args.data_dir = str(data_dir)


    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Actor-Critic Agent Training for 2048 Game")
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
    print("\nCreating Actor-Critic Agent...")
    agent = ActorCriticAgent(
        special_pos=special_pos,
        auto_detect_special=auto_detect,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_critic=args.learning_rate_critic,
        gamma=args.gamma,
        hidden_size=args.hidden_size
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
    print(f"  Actor learning rate: {args.learning_rate_actor}")
    print(f"  Critic learning rate: {args.learning_rate_critic}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Device: {agent.device}")
    print(f"  Use collected data: {args.use_collected_data}")
    print("\n" + "=" * 60)
    
    try:
        if args.use_collected_data:
            # 使用 collected_data 训练
            train_from_collected_data(
                agent,
                data_dir=args.data_dir,
                num_epochs=args.num_epochs
            )
            # 保存最终模型
            agent.save(args.save_path)
        else:
            # 自对弈训练
            episode_rewards, episode_lengths = train_actor_critic_agent(
                agent,
                num_episodes=args.episodes,
                save_freq=args.save_freq,
                save_path=args.save_path
            )
            
            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print(f"Final model saved to: {args.save_path}")
            if len(episode_rewards) > 0:
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
        import traceback
        traceback.print_exc()
        print(f"Attempting to save current model to {args.save_path}...")
        try:
            agent.save(args.save_path)
            print("Model saved.")
        except:
            print("Failed to save model.")


if __name__ == "__main__":
    main()
