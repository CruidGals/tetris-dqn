import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from replay_buffer import ExperienceReplay
from mdp.env import get_possible_obs

class AfterstateValueModel(nn.Module):
    def __init__(self, input):
        super(AfterstateValueModel, self).__init__()

        self.net = nn.Sequential(nn.Linear(input, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))
        
    def forward(self, X):
        return self.net(X).squeeze(-1)
    
class AfterstateValueNetwork:
    def __init__(self, input, replay_buffer_cap, min_buffer_before_training, target_update, epsilon_start=1.0, epsilon_end = 1e-3, temperature=0.99, lr=1e-3, gamma=0.99, weight_decay = 1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model updated every time, target model update every couple steps
        self.model = AfterstateValueModel(input).to(self.device)

        self.target_model = AfterstateValueModel(input).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_update = target_update
        self.target_model.eval()

        self.replay_buffer = ExperienceReplay(replay_buffer_cap)
        self.min_buffer_before_training = min_buffer_before_training

        # Q network params
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.temperature = temperature

        # Network params
        self.loss = F.mse_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loggers
        self.writer = SummaryWriter()
        self.training_step = 0

    def remember(self, grid_after, obs_after, reward, next_block, done):
        """ Appends transition onto the replay memory """
        self.replay_buffer.add(grid_after, obs_after, reward, next_block, done)

    def act(self, poss_obs: np.ndarray):
        """ 
        Given possible observations, give the best next action for the state.
        
        Depending on a random value, explore if value is within epsilon and exploit when not
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(40)

        poss_values = self._possible_values(poss_obs)

        # Return the best action from the values
        return poss_values.argmax().item()

    # Overloaded functions
    def _possible_values(self, poss_obs: np.ndarray) -> torch.Tensor:
        """ Given possible observations, give the predicted value for each observation """
        with torch.no_grad():
            poss_obs_tensor = torch.as_tensor(poss_obs, dtype=torch.float32, device=self.device)
        return self.model(poss_obs_tensor)
    
    # Training and updating
    def train(self):
        """ Sample transitions from the replay buffer and learn """

        if len(self.replay_buffer) < self.min_buffer_before_training:
            return
        
        # Get a batch of transitions and store them into tensors
        batch = self.replay_buffer.sample(self.batch_size)
        grid_after, obs_after, rewards, next_block, dones = zip(*batch)

        # Convert them into tensors (except next_block)
        obs_after = torch.FloatTensor(np.array(obs_after), device=self.device)
        rewards = torch.FloatTensor(np.array(rewards), device=self.device)
        dones = torch.BoolTensor(np.array(dones), device=self.device)

        # Predicted values
        curr_pred = self.model(obs_after)

        # Go through batches and make target_pred
        target_pred = []
        with torch.no_grad():
            for i in range(len(batch)):
                if dones[i]:
                    target_pred.append(rewards[i])
                else:
                    poss_obs = get_possible_obs(grid_after, next_block)
                    poss_vals = self._possible_values(poss_obs)
                    a_star = int(torch.argmax(poss_vals))
                    target_vals = self.target_model(torch.as_tensor(poss_vals, dtype=torch.float32, device=self.device))
                    boot = target_vals[a_star]
                    target_pred.append(rewards[i] + self.gamma * boot)

        target_pred = torch.stack(target_pred)

        # Compute loss
        loss = self.loss(curr_pred, target_pred)

        # Log loss
        self.writer.add_scalar('Loss/train', loss.item(), self.training_step)
        
        # Backpropogate
        self.optimizer.zero_grad()
        loss.backward()

        # Write gradient to writer
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_scalar(f'GradNorm/{name}', param.grad.norm(), self.training_step)

        self.optimizer.step()
        self.training_step += 1

        # Update the parameters
        self._update()

        return loss.item()
    
    def _update(self):
        """ Updates the training model and decreases epsilon """
        if self.training_step % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)

    # Saving and loading model
    def save(self, path):
        torch.save(self.target_model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())