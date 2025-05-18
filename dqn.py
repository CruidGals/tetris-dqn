import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.optim import Adam
from model import DQNModel
from collections import deque

REPLAY_MEMORY_SIZE = 50_000

class DQNAgent:
    def __init__(self, input=200, output=4, action_size=5, learning_rate=1e-4, gamma=0.99, batch_size=256, 
                 target_update=5, epsilon=1.0, epsilon_min=0.1, decay_rate=0.995, device='cpu'):
        self.device = torch.device("cuda" if device == 'cuda' and torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Epsilon Decay information
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate

        # Main model that gets trained every step
        self.model = self.create_model(input, output).to(self.device)

        # Target model updated every C steps, but is used for prediction against the actual model
        self.target_model = self.create_model(input, output).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Instantiate replay memory
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # NN stuff
        self.optimizer = Adam(self.model.parameters())
        self.loss = F.mse_loss

    def create_model(self, input, output) -> DQNModel:
        return DQNModel(input, output)
    
    def update(self):
        # Update the target network with epsilon too
        self.target_model.load_state_dict(self.model.state_dict())

        # Do epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

    def update_replay_memory(self, transition):
        """
        Update the replay memory of the agent. If the replay memory is full, the transition is put onto the queue
        and the latest memory gets tossed.

        Transition consists of (state, action, reward, next_state, done)
        """
        self.replay_memory.append(transition)

    def act(self, state):
        """
        Get the Q-value based on the weights of the model
        """

        self.model.eval()
        result = None

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            result = self.model(state_tensor).argmax().item()
        
        self.model.train()
        return result
    
    def train(self):
        """
        Train the policy network
        """
        if len(self.replay_memory) < self.batch_size:
            return
        
        # Get a random batch and store them into tensors
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        # The "dones" array contains booleans that indicate whether if the next state stops the episode
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        curr_q_vals = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_q_vals = self.target_model(next_states).max(1, keepdim=True)[0]
            target_q_vals = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_vals
        
        loss = self.loss(curr_q_vals, target_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss