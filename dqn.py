import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from model import DQNModel
from collections import deque

class DQNAgent:
    def __init__(self, input=200, output=5, action_size=5, learning_rate=1e-4, weight_decay=1e-3, gamma=0.99, batch_size=256, replay_buffer_size=100000,
                 min_buffer_before_training=5000, target_update=10, epsilon=1.0, epsilon_min=0.1, lr_gamma=0.5, lr_step_size=400, lr_end=0.00001, decay_rate=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
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
        self.target_update = target_update
        self.target_update_counter = 0

        # Instantiate replay memory
        self.replay_memory = deque(maxlen=replay_buffer_size)
        self.min_buffer_before_training = min_buffer_before_training

        # NN stuff
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss = F.mse_loss
        self.lr_scheduler = StepLR(self.optimizer, lr_step_size, lr_gamma)
        self.lr_end = lr_end

        # Tensorboard
        self.writer = SummaryWriter()
        self.training_step = 0

    def create_model(self, input, output) -> DQNModel:
        return DQNModel(input, output)
    
    def update(self):
        self.update_target_model()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

    def update_target_model(self):
        # Make sure to update once hit target update
        if self.target_update_counter < self.target_update:
            self.target_update_counter += 1
            return

        self.target_update_counter = 0

        # Update the target network with epsilon too
        self.target_model.load_state_dict(self.model.state_dict())


    def remember(self, transition):
        """
        Update the replay memory of the agent. If the replay memory is full, the transition is put onto the queue
        and the latest memory gets tossed. Also updates the recent rewards for epsilon decay

        Transition consists of (state, action, reward, next_state, done)
        """
        self.replay_memory.append(transition)

    def act(self, state):
        """
        Get the Q-value based on the weights of the model
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.model(state_tensor).argmax().item()
    
    def train(self):
        """
        Train the policy network
        """
        if len(self.replay_memory) < self.min_buffer_before_training:
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

        # Write the loss to writer
        self.writer.add_scalar('Loss/train', loss, self.training_step)

        self.optimizer.zero_grad()
        loss.backward()

        # Write gradient to writer
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_scalar(f'GradNorm/{name}', param.grad.norm(), self.training_step)

        self.optimizer.step()

        # Update learning rate
        if self.lr_scheduler.get_lr()[0] > self.lr_end:
            self.lr_scheduler.step()

        # Increment x-axis for logs
        self.training_step += 1

        return loss
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())