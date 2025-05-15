import yaml
import argparse
import time
from dqn import DQNAgent
from tetris_env import Tetris

def train(config):
    """
    Train the agent based on the hyperparameters
    """
    agent = DQNAgent(
        input=config['network']['input'],
        output=config['network']['output'],
        action_size=config['agent']['action_size'],
        learning_rate=config['agent']['learning_rate'],
        gamma=config['agent']['gamma'],
        batch_size=config['agent']['batch_size'],
        target_update=config['agent']['target_update_rate'],
        epsilon=config['agent']['epsilon'],
        epsilon_min=config['agent']['epsilon_min'],
        decay_rate=config['agent']['decay_rate'],
    )

    env = Tetris()

    # Begin training
    for i in range(config['training']['episodes']):
        state = env.reset()
        start_time = time.time()
        curr_time = start_time
        max_time = config['training']['max_episode_length']
        

def eval(config, agent, env):
    pass

def hyperparams_parser(args):
    with open(args, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    agent, env = train(config)

# Run the DQN episodically
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="Path to the yaml file for hyperparamaters")
    args = parser.parse_args()

    hyperparams_parser(args.input)