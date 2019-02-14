import torch
import numpy as np


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FullyConv_LSTM:

    def __init__(self):
        super(FullyConv_LSTM, self).__init__()

        self.MinimapNet = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
        )

        self.ScreenNet = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
        )

        self.SpatialPolicy = torch.nn.Conv2d(32+32+0, 1, 1)

        self.LSTM = torch.nn.LSTMCell(0, 256)

        self.Value = torch.nn.Linear(256, 1)
        self.NonSpatialPolicy = torch.nn.Linear(256, 0)

    def forward(self, screen, minimap, features, x):

        scr_data = self.ScreenNet(screen)
        map_data = self.MinimapNet(minimap)

        # spatialize features and afterwards concatenate
        conc_data = None

        spatial_policy = self.SpatialPolicy(conc_data)

        nonspatial_data, cx = self.LSTM(Flatten(conc_data), x)
        nonspatial_policy = self.NonSpatialPolicy(nonspatial_data)
        value = self.Value(nonspatial_data)

        return spatial_policy, nonspatial_policy, value, (nonspatial_data, cx)




class ActorCriticModel_SpaceInvaders(torch.nn.Module):

    def __init__(self):
        super(ActorCriticModel_SpaceInvaders, self).__init__()

        self.ActorCritic = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=1),
            torch.nn.ReLU(),
            Flatten(),
        )

        self.LSTM = torch.nn.LSTMCell(7 * 6 * 32, 256)

        self.Policy = torch.nn.Linear(256, 6)
        self.Value = torch.nn.Linear(256, 1)


    def forward(self, data, x):

        data = self.ActorCritic(data)

        hx, cx = self.LSTM(data, x)
        data = hx

        Logit = self.Policy(data)
        Value = self.Value(data)
        return Logit, Value, (hx, cx)