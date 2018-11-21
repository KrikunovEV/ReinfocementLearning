import torch


def normalized_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


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
            torch.nn.Conv2d(32, 64, 4, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
            Flatten(),
            torch.nn.Linear(3 * 4 * 64, 512),
            torch.nn.ReLU()
        )

        self.Policy = torch.nn.Linear(512, 6)
        self.Policy.weight.data = normalized_initializer(self.Policy.weight.data, 0.01)
        self.Policy.bias.data.fill_(0)

        self.Value = torch.nn.Linear(512, 1)
        self.Value.weight.data = normalized_initializer(self.Value.weight.data, 1.0)
        self.Value.bias.data.fill_(0)


    def forward(self, input):
        data = self.ActorCritic(input)
        Logit = self.Policy(data)
        Value = self.Value(data)
        return Logit, Value