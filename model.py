import torch
import torch.nn as nn

class DQNModel(nn.Module):

    def __init__(self, input, output):
        super(DQNModel, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(input, input, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),
                                 nn.LazyConv2d(input // 2, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2, 2),
                                 nn.Flatten(),
                                 nn.LazyLinear(512), nn.RelU(), nn.Dropout(0.2),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, output))
        
        self.net.apply(init_params)

    def forward(self, X):
        return self.net(X)
        
def init_params(module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.LazyConv2d or type(module) == nn.LazyLinear:
            nn.init.normal_(module.weight)