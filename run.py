import yaml
import argparse
import time
import numpy as np
import os
from dqn import DQNAgent
from tetris_env import Tetris

def train(config):
    """
    Train the agent based on the hyperparameters
    """
    timestamp = time.strftime('%H%d%m%Y')
    result_dir = os.path.join('results/models', f"experiment_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Save config file
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    agent = DQNAgent(
        input=config['network']['input'],
        output=config['network']['output'],
        action_size=config['agent']['action_size'],
        learning_rate=config['agent']['learning_rate'],
        weight_decay=config['agent']['weight_decay'],
        gamma=config['agent']['gamma'],
        batch_size=config['agent']['batch_size'],
        replay_buffer_size=config['agent']['replay_buffer_size'],
        min_buffer_before_training=config['agent']['min_buffer_before_training'],
        target_update=config['agent']['target_update_rate'],
        epsilon=config['agent']['epsilon'],
        epsilon_min=config['agent']['epsilon_min'],
        lr_gamma=config['agent']['lr_scheduler']['gamma'],
        lr_step_size=config['agent']['lr_scheduler']['step_size'],
        lr_end=config['agent']['lr_scheduler']['lr_end'],
        decay_rate=config['agent']['decay_rate']
    )

    env = Tetris(use_pygame=config['training']['render'])
    best_reward = -np.inf
    best_episode = 0
    best_episode_replay = None

    # Begin training
    for i in range(config['training']['episodes']):
        state = env.reset()
        total_reward = 0
        ep_loss = 0

        # Keep track of the states in this episode
        episode_replay = [state]

        # Accumulate rewards until landed
        reward = 0

        # Run episode until max length of ~30 seconds
        start_time = time.time()
        prev_time = start_time
        max_time = config['training']['max_episode_length']

        while prev_time - start_time < max_time:
            time.sleep(0.001)
            
            # Update current_time
            prev_time = time.time()

            # Start the training process
            action = agent.act(state)
            next_state = env.move(action)

            # Store state in replay
            episode_replay.append(next_state)

            # Falling behavior (every 5 frames)
            if env.clock.frame_count % 4 == 0:
                block_landed = env.fall()

            reward += env.distribute_reward()
            env.clear_rows()
            done = env.done
            
            if block_landed:
                agent.remember((state, action, reward, next_state, done))
                total_reward += reward
                reward = 0

            if config['training']['render']:
                env.render()

            if done:
                break

            # Train less frequently
            if env.clock.frame_count % 4 == 0:
                loss = agent.train()
                ep_loss += loss if loss else 0
            
            state = next_state

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = i + 1
            best_episode_replay = episode_replay

        # Update the epsilon and critic model after every episode
        agent.update()

        print(f"Episode: {i+1}; Reward: {total_reward:.3f}; Epsilon: {agent.epsilon:.2f}; Landed: {env.landed_block_count}; Loss/per: {(ep_loss / env.landed_block_count):.2f}; Cleared: {env.rows_cleared}; Buffer: {len(agent.replay_memory)}")
        
        # Save agent and episode replay after every 500 episodes
        if (i+1) % 500 == 0:
            agent.save(os.path.join(result_dir, f"model_{i+1}.pth"))
            np.save(os.path.join(result_dir, f'episode_replay_{i + 1}.npy'), episode_replay)
    
    print(f"Best reward: {best_reward}; Best episode: {best_episode}")
    np.save(os.path.join(result_dir, f'best_episode_replay_{best_episode}.npy'), best_episode_replay)

    # Save the overall agent
    agent.save(os.path.join(result_dir, f"model_{config['training']['episodes']}.pth"))

    return agent, env


def eval(config, agent, env):
    pass

def run(args):
    with open(args, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    agent, env = train(config)
    print('Training completed')
    # TODO eval
    env.close()

# Run the DQN episodically
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='hyperparams.yaml', help="Path to the yaml file for hyperparamaters")
    args = parser.parse_args()

    run(args.input)