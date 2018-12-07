import torch


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ActorCriticModel(torch.nn.Module):

    def __init__(self):
        super(ActorCriticModel, self).__init__()

        self.ActorCritic = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=1),
            torch.nn.ReLU(),
            Flatten(),
            torch.nn.Linear(7 * 6 * 32, 512),
            torch.nn.ReLU()
        )

        self.Policy = torch.nn.Linear(512, 6, bias=False)
        self.Value = torch.nn.Linear(512, 1, bias=False)


    def forward(self, input):
        data = self.ActorCritic(input)
        Logit = self.Policy(data)
        Value = self.Value(data)
        return Logit, Value

    def getActorParameters(self):
        return (list(self.ActorCritic.parameters()) + list(self.Policy.parameters()))

    def getCriticParameters(self):
        return self.Value.parameters()