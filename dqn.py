import torch
import torch.nn as nn
import numpy as np
from model import DQNModel
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPLAY_MEMORY_SIZE = 50_000

class DQNAgent:
    def __init__(self, input, output):

        # Main model that gets trained every step
        self.model = self.create_model(input, output)

        # Target model updated every C steps, but is used for prediction against the actual model
        self.target_model = self.create_model(input, output)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_update_counter = 0

        # Instantiate replay memory
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def create_model(self, input, output) -> DQNModel:
        return DQNModel(input, output)
    
    def update_replay_memory(self, transition):
        """
        Update the replay memory of the agent. If the replay memory is full, the transition is put onto the queue
        and the latest memory gets tossed.
        """
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        """
        Get the Q-values based on the weights of the model
        """

        self.model.eval()
        result = None

        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array(state).reshape(-1, *state.shape) / 255).to(device)

            # Get the predicted q value
            result = self.model(state_tensor)[0]
            print(f'Step {step}: Predicted Q-Values: {result}')
        
        self.model.train()
        return result