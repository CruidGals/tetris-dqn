import yaml
import argparse
import time
import numpy as np
import os
from afterstate_model import AfterstateValueNetwork
from tetris_env import TetrisEnv
from mdp.env import get_possible_obs

def train(config):
    """
    Train the agent based on the hyperparameters
    """
    timestamp = time.strftime('%Y%m%d%H')
    result_dir = os.path.join('results/models', f"{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Save config file
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    agent = AfterstateValueNetwork(
        input=config['network']['input'],
        batch_size=config['agent']['batch_size'],
        replay_buffer_cap=config['agent']['replay_buffer_size'],
        min_buffer_before_training=config['agent']['min_buffer_before_training'],
        target_update=config['agent']['target_update_rate'],
        epsilon_start=config['agent']['epsilon'],
        epsilon_end=config['agent']['epsilon_min'],
        temperature=config['agent']['decay_rate'],
        lr=config['agent']['learning_rate'],
        gamma=config['agent']['gamma'],
        weight_decay=config['agent']['weight_decay']
    )

    env = TetrisEnv(headless=(not config['training']['render']))
    best_reward = -np.inf
    best_episode = 0
    best_episode_replay = None

    # Begin training
    for i in range(config['training']['episodes']):
        # Initialize episode
        state = env.reset()
        total_reward = 0
        ep_loss = 0

        # Keep track of the states in this episode (replay is the first 200 dims)
        episode_replay = [env.grid.copy()]

        # Run episode until max length of ~30 seconds
        start_time = time.time()
        prev_time = start_time
        max_time = config['training']['max_episode_length']

        while prev_time - start_time < max_time:
            # time.sleep(1 / 480)
            
            # Update current_time
            prev_time = time.time()

            # Start the training process
            action = agent.act(get_possible_obs(env.grid, env.controlled_block))

            # Capture step the env:
            obs, rew, term = env.step(action)
            episode_replay.append(env.grid.copy())
            agent.remember(env.grid.copy(), obs, rew, env.controlled_block, term)
            total_reward += rew

            if config['training']['render']:
                env.render()

            if term:
                break
            
            state = obs

        # Save the best replay
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = i + 1
            best_episode_replay = episode_replay

        # Do 5 training bursts after every episode
        for _ in range(5):
            loss = agent.train()
            ep_loss += loss if loss is not None else 0

        # Print episode information
        print(f"Episode: {i+1}; Avg Reward: {(total_reward / env.landed_block_count):.3f}; Epsilon: {agent.epsilon:.3f}; Landed: {env.landed_block_count}; Loss/per: {(ep_loss / env.landed_block_count):.2f}; Cleared: {env.rows_cleared}")
        
        # Save agent and episode replay after every 500 episodes
        if (i+1) % 500 == 0:
            agent.save(os.path.join(result_dir, f"model_{i+1}.pth"))
            np.save(os.path.join(result_dir, f'episode_replay_{i + 1}.npy'), episode_replay)
    
    print(f"Best reward: {best_reward}; Best episode: {best_episode}")
    np.save(os.path.join(result_dir, f'best_episode_replay_{best_episode}.npy'), best_episode_replay)

    # Save the overall agent
    agent.save(os.path.join(result_dir, f"model_{config['training']['episodes']}.pth"))

    return agent, env


def test(config, agent: AfterstateValueNetwork):
    """
    Test the agent on a stable environment
    """
    timestamp = time.strftime('%Y%m%d%H')
    result_dir = os.path.join('results/eval', f"{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Save config file
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    env = TetrisEnv(headless=not config['evaluation']['render'])
    agent.epsilon = 0.0

    # Saving best episodes and rewards
    best_reward = -np.inf
    best_episode = 0
    best_episode_replay = None

    print("Beginning evaluation...")

    for i in range(config['evaluation']['episodes']):
        state = env.reset()
        total_reward = 0

        start_time = time.time()
        prev_time = start_time
        max_time = config['training']['max_episode_length']

        # Episode replay
        episode_replay = [env.grid.copy()]

        while prev_time - start_time < max_time:
            prev_time = time.time()

            action = agent.act(get_possible_obs(env.grid, env.controlled_block))
            obs, rew, term = env.step(action)
            episode_replay.append(env.grid.copy())
            total_reward += rew

            if config['evaluation']['render']:
                env.render()

            if term:
                break
            
            state = obs

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = i + 1
            best_episode_replay = episode_replay

        print(f"Test Episode: {i+1}; Avg Reward: {(total_reward / env.landed_block_count):.3f}; Landed: {env.landed_block_count}; Cleared: {env.rows_cleared}")
    
    print(f"Best reward: {best_reward}; Best episode: {best_episode}")
    np.save(os.path.join(result_dir, f'best_episode_replay_{best_episode}.npy'), best_episode_replay)

    env.close()

def run(args):
    with open(args, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    agent, env = train(config)
    print('Training completed')
    env.close()

def eval(args):
    with open(args, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    agent = AfterstateValueNetwork(
        input=config['network']['input'],
        batch_size=config['agent']['batch_size'],
        replay_buffer_cap=config['agent']['replay_buffer_size'],
        min_buffer_before_training=config['agent']['min_buffer_before_training'],
        target_update=config['agent']['target_update_rate'],
        epsilon_start=0.0,
        epsilon_end=0.0,
        temperature=1.0,
        lr=config['agent']['learning_rate'],
        gamma=config['agent']['gamma'],
        weight_decay=config['agent']['weight_decay']
    )

    # Load the model
    agent.load(config['evaluation']['model_path'])
    print('Model loaded!')

    test(config, agent)

# Run the DQN episodically
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='src/hyperparams.yaml', help="Path to the yaml file for hyperparamaters")
    parser.add_argument('-e', '--eval', type=bool, default=False, help="Run evaluation instead of training")
    args = parser.parse_args()

    run(args.input) if not args.eval else eval(args.input)