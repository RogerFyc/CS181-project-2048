
def main():
    parser = argparse.ArgumentParser(description='Train DQN Agent for 2048')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of training episodes (default: 10000)')
    parser.add_argument('--save-freq', type=int, default=1