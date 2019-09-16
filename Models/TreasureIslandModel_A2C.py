import torch
import torch.nn as nn


class TImodel_A2C(nn.Module):

    def __init__(self, feature_size):
        super().__init__()

        self.feature_size = feature_size

        self.preproc = nn.Conv2d(4, 2, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
        )

        self.flat = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU()
        )

        self.policy = nn.Linear(256, 6)
        self.value = nn.Linear(256, 1)

    def forward(self, obs):
        scalar_features = torch.Tensor(obs.treasure_map.data).view(1, 1, self.feature_size[0], self.feature_size[1])
        categorical_features = self._preprocess([obs.entity_map, obs.cell_map])

        obs = torch.cat((categorical_features, scalar_features), dim=1)

        data = self.conv(obs).flatten()
        data = self.flat(data)
        return self.policy(data), self.value(data)

    def _preprocess(self, features):
        categorical = torch.Tensor()

        for feature in features:
            tmp = torch.LongTensor(feature.data).view(1, 1, self.feature_size[0], self.feature_size[1])
            one_hots = torch.Tensor(tmp.size(0), feature.scale, tmp.size(2), tmp.size(3)).zero_()
            one_hots = one_hots.scatter_(1, tmp.data, 1)
            categorical = torch.cat((categorical, one_hots[0]))

        return self.preproc(categorical.unsqueeze(0))
