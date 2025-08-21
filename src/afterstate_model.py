import torch
import torch.nn as nn
import numpy as np
from replay_buffer import PERBuffer

class AfterstateValueModel(nn.Module):
    def __init__(self, input):
        super(AfterstateValueModel, self).__init__()

        self.net = nn.Sequential(nn.Linear(input, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))
        
    def forward(self, X):
        return self.net(X)
    
class AfterstateValueNetwork:
    def __init__(self, input, replay_buffer_cap, min_buffer_before_training):
        self.model = AfterstateValueModel(input)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = PERBuffer(replay_buffer_cap)
        self.min_buffer_before_training = min_buffer_before_training

    def act(self, poss_obs: np.ndarray):
        """ Given possible observations, give the best next action for the state"""
        poss_values = self._possible_values(poss_obs)

        # Return the best action from the values
        return poss_values.argmax().item()

    def _possible_values(self, poss_obs: np.ndarray) -> torch.Tensor:
        """ Given possible observations, give the predicted value for each observation """
        poss_obs_tensor = torch.tensor(poss_obs, device=self.device)
        return self.model(poss_obs_tensor)
    
    def learn(self):
        """ Sample transitions from the replay buffer and learn """

        if len(self.replay_buffer) < self.min_buffer_before_training:
            return
        
        # Get a batch of transitions and store them into tensors
        batch, idx, is_weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to GPU-accelerated tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(np.array(is_weights)).unsqueeze(1).to(self.device) # Importance sampling weights

        