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
            torch.nn.Conv2d(1, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ELU(),
            Flatten(),
            torch.nn.Linear(9 * 6 * 64 * 10, 256),
            torch.nn.ELU()
        )

        self.Policy = torch.nn.Linear(256, 6)
        self.Policy.weight.data = normalized_initializer(self.Policy.weight.data, 0.01)
        self.Policy.bias.data.fill_(0)

        self.Value = torch.nn.Linear(256, 1)
        self.Value.weight.data = normalized_initializer(self.Value.weight.data, 1.0)
        self.Value.bias.data.fill_(0)


    def forward(self, input):
        data = self.ActorCritic(input)
        Logit = self.Policy(data)
        Value = self.Value(data)
        return Logit, Value