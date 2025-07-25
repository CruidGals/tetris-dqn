import torch.nn as nn

class DQNModel(nn.Module):

    def __init__(self, input, output):
        super(DQNModel, self).__init__()

        self.net = nn.Sequential(nn.Linear(input, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, output))

        # self.net = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
        #                          nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),     
        #                          nn.Flatten(),
        #                          nn.Linear(64 * input, 128), nn.ReLU(),
        #                          nn.Linear(128, output))
        
        self.net.apply(init_params)

    def forward(self, X):
        return self.net(X)
        
def init_params(module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.LazyConv2d or type(module) == nn.LazyLinear:
            nn.init.normal_(module.weight)