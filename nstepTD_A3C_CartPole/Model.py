import torch

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()


        self.ActorCritic = torch.nn.Sequential(
            torch.nn.Linear(4, 24),
            torch.nn.ReLU(),
        )

        self.Actor = torch.nn.Sequential(
            torch.nn.Linear(24, 2)
        )

        self.Critic = torch.nn.Sequential(
            torch.nn.Linear(24, 1)
        )


    def forward(self, input):
        data = self.ActorCritic(input)
        Logit = self.Actor(data)
        Value = self.Critic(data)
        return Logit, Value


    def CriticParameters(self):
        return self.Critic.parameters()


    def ActorParameters(self):
        return list(self.ActorCritic.parameters()) + list(self.Actor.parameters())