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
            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            Flatten(),
        )

        self.LSTM = torch.nn.LSTMCell(32 * 5 * 7, 256)
        self.LSTM.bias_ih.data.fill_(0)
        self.LSTM.bias_hh.data.fill_(0)

        self.Policy = torch.nn.Linear(256, 6)
        self.Policy.weight.data = normalized_initializer(self.Policy.weight.data, 0.01)
        self.Policy.bias.data.fill_(0)

        self.Value = torch.nn.Linear(256, 1)
        self.Value.weight.data = normalized_initializer(self.Value.weight.data, 1.0)
        self.Value.bias.data.fill_(0)


    def forward(self, input):
        data, (hx, cx) = input
        data = self.ActorCritic(data)

        hx, cx = self.LSTM(data, (hx, cx))

        x = hx
        Logit = self.Policy(x)
        Value = self.Value(x)
        return Logit, Value, (hx, cx)