import torch

from Util import *

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FullyConv_LSTM(torch.nn.Module):

    def __init__(self):
        super(FullyConv_LSTM, self).__init__()

        self.MinimapNet = torch.nn.Sequential(
            torch.nn.Conv2d(FeatureMinimapCount, 16, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU()
        )

        self.ScreenNet = torch.nn.Sequential(
            torch.nn.Conv2d(FeatureScrCount, 16, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU()
        )

        self.SpatialPolicy = torch.nn.Conv2d(32 + 32, 1, 1)

        # features size**2, 32 filters, 2 from minimap and screen nets
        self.Linear = torch.nn.Sequential(
            torch.nn.Linear(Hyperparam["FeatureSize"]**2 * 32 * 2, 256),
            torch.nn.ReLU()
        )

        self.Value = torch.nn.Linear(256, 1)
        self.Policy = torch.nn.Linear(256, FunctionCount)


    def forward(self, screens, minimaps, flat):
        scr_data = self.ScreenNet(screens)
        map_data = self.MinimapNet(minimaps)

        # How to spatialize flat features ?
        conc_data = torch.cat((scr_data, map_data), 0)

        spatial_logits = self.SpatialPolicy(conc_data)

        nonspatial_data = self.Linear(Flatten(conc_data))
        logits = self.Policy(nonspatial_data)
        value = self.Value(nonspatial_data)

        return spatial_logits, logits, value