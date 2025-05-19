import yaml
import argparse
import time
import numpy as np
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
        decay_rate=config['agent']['decay_rate']
    )

    env = Tetris(use_pygame=config['training']['render'])
    best_reward = -np.inf

    # Begin training
    for i in range(config['training']['episodes']):
        state = env.reset()
        total_reward = 0
        ep_loss = 0

        # Run episode until max length of ~30 seconds
        start_time = time.time()
        prev_time = start_time
        max_time = config['training']['max_episode_length']

        while prev_time - start_time < max_time:
            curr_time = time.time()

            # Make a move every millisecond
            if curr_time - prev_time < 1/1000:
                continue
            
            # Update current_time
            prev_time = curr_time

            # Start the training process
            reward = 0
            action = agent.act(state)
            next_state = env.move(action)

            # Falling behavior (every 5 frames)
            if env.clock.frame_count % 4 == 0:
                reward += env.fall()

            total_reward += reward

            done = env.done
            agent.remember((state, action, reward, next_state, done))

            # Train less frequently
            if env.clock.frame_count % 4 == 0:
                loss = agent.train()
                ep_loss += loss if loss else 0
            
            state = next_state

            if config['training']['render']:
                env.render()

            if done:
                break

        best_reward = max(best_reward, total_reward)

        # Update the epsilon and critic model after every episode
        agent.update()
        print(f"Episode: {i}; Reward: {total_reward:.3f}; Epsilon: {agent.epsilon:.2f}; Landed: {env.landed_block_count}; Loss/per: {(ep_loss / env.landed_block_count):.2f}")

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