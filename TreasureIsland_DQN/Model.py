import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU()
        )

        self.flat = nn.Linear(32, 6)

    def forward(self, data):
        data = self.conv(data)
        data = data.view(data.size(0), -1)
        data = self.flat(data)
        return data
