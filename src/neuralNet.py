import torch
import torch.nn as nn
from config import windowLength

class MovementNet(nn.Module):
    def __init__(self):
        super(MovementNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6 * windowLength, 64),   # 6 input features
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)    # 2 output values: dx, dy
        )

    def forward(self, x):
        return self.model(x)