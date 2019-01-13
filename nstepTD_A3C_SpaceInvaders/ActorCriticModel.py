import torch
import numpy as np


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)



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

        #self.apply(weights_init)

        #self.Policy.weight.data = normalized_columns_initializer(self.Policy.weight.data, 0.01)
        #self.Value.weight.data = normalized_columns_initializer(self.Value.weight.data, 1.0)

        #self.LSTM.bias_hh.data.fill_(0)
        #self.LSTM.bias_ih.data.fill_(0)

        #self.Policy.bias.data.fill_(0)
        #self.Value.bias.data.fill_(0)

        #self.train()


    def forward(self, data, x):

        data = self.ActorCritic(data)

        hx, cx = self.LSTM(data, x)
        data = hx

        Logit = self.Policy(data)
        Value = self.Value(data)
        return Logit, Value, (hx, cx)

    def getActorParameters(self):
        return (list(self.ActorCritic.parameters()) + list(self.LSTM.parameters()) + list(self.Policy.parameters()))

    def getCriticParameters(self):
        return self.Value.parameters()


class ActorCriticModel_Breakout(torch.nn.Module):

    def __init__(self):
        super(ActorCriticModel_Breakout, self).__init__()

        self.ActorCritic = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=1),
            torch.nn.ReLU(),
            Flatten(),
        )

        self.LSTM = torch.nn.LSTMCell(5 * 6 * 32, 256)

        self.Policy = torch.nn.Linear(256, 4)
        self.Value = torch.nn.Linear(256, 1)

        self.apply(weights_init)

        self.Policy.weight.data = normalized_columns_initializer(self.Policy.weight.data, 0.01)
        self.Value.weight.data = normalized_columns_initializer(self.Value.weight.data, 1.0)

        self.LSTM.bias_hh.data.fill_(0)
        self.LSTM.bias_ih.data.fill_(0)

        self.Policy.bias.data.fill_(0)
        self.Value.bias.data.fill_(0)

        self.train()


    def forward(self, data, x):

        data = self.ActorCritic(data)

        hx, cx = self.LSTM(data, x)
        data = hx

        Logit = self.Policy(data)
        Value = self.Value(data)
        return Logit, Value, (hx, cx)

    def getActorParameters(self):
        return list(self.ActorCritic.parameters()) + list(self.LSTM.parameters()) + list(self.Policy.parameters())

    def getCriticParameters(self):
        return self.Value.parameters()