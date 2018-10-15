import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU()
        ).cuda()

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(9 * 6 * 64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 6)
        ).cuda()

    def forward(self, data):
        data = self.conv_layers(data)
        data = data.view(len(data), 9 * 6 * 64)
        data = self.fc_layers(data)
        return data

    def noGradForward(self, data):
        with torch.no_grad():
            data = self.conv_layers(data)
            data = data.view(len(data), 9 * 6 * 64)
            data = self.fc_layers(data)
        return data