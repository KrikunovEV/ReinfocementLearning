import torch
import numpy as np

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ActorCriticModel(torch.nn.Module):

    def __init__(self):
        super(ActorCriticModel, self).__init__()

        self.ActorCritic = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
            Flatten(),
            torch.nn.Linear(9 * 6 * 64, 512),
            torch.nn.ReLU()
        )


        self.Policy = torch.nn.Linear(512, 6)
        self.Policy.bias.data.fill_(0)


        self.Value = torch.nn.Linear(512, 1)
        self.Value.bias.data.fill_(0)


    def forward(self, input):
        data = self.ActorCritic(input)
        Logit = self.Policy(data)
        Value = self.Value(data)
        return Logit, Value