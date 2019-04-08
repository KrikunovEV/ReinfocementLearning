import torch
import torch.nn as nn
from Util import *


class FullyConv(nn.Module):

    def __init__(self):
        super(FullyConv, self).__init__()

        self.MapPreproc = nn.Conv2d(Params["MapPreprocNum"], FeatureMinimapCount, 1)
        self.ScrPreproc = nn.Conv2d(Params["ScrPreprocNum"], FeatureScrCount - 2, 1) # -2 for SCALAR

        self.FlatNet = nn.Sequential(
            nn.Linear(11, Params["FeatureSize"]**2),
            nn.Tanh()
        )

        self.MinimapNet = nn.Sequential(
            nn.Conv2d(FeatureMinimapCount, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.ScreenNet = nn.Sequential(
            nn.Conv2d(FeatureScrCount, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.SpatialPolicy = nn.Conv2d(32 + 32 + 1, 1, 1)

        # features size**2, 32 filters, 2 from minimap and screen nets + from flat features
        self.LinearNet = nn.Sequential(
            nn.Linear(Params["FeatureSize"]**2 * (32 * 2 + 1), 256),
            nn.ReLU()
        )

        self.Value = nn.Linear(256, 1)
        self.Policy = nn.Linear(256, FunctionCount)

    def forward(self, scr_features, map_features, flat_features):

        scr_features = self._preprocess(scr_features, Type.SCREEN)
        map_features = self._preprocess(map_features, Type.MINIMAP)
        flat_features = self._preprocess(flat_features, Type.FLAT)

        scr_features = self.ScreenNet(scr_features)
        map_features = self.MinimapNet(map_features)

        features = torch.cat((scr_features, map_features, flat_features), 1)

        spatial_logits = self.SpatialPolicy(features)

        nonspatials = features.flatten()
        nonspatials = self.LinearNet(nonspatials)
        logits = self.Policy(nonspatials)
        value = self.Value(nonspatials)

        return spatial_logits, logits, value

    def _preprocess(self, features, features_type):

        if features_type == Type.FLAT:
            features = self.FlatNet(torch.Tensor(features).cuda()).view(1, 1, Params["FeatureSize"], Params["FeatureSize"])

        elif features_type == Type.SCREEN:
            categorical = torch.Tensor().cuda()
            numerical = torch.Tensor().cuda()

            for i, feature in enumerate(features):
                if SCREEN_FEATURES[i].type == CATEGORICAL:

                    scale = SCREEN_FEATURES[i].scale
                    if i == 1:
                        scale = len(MY_UNIT_TYPE) + 1
                        feature = np.array(feature)
                        for j, unit_type in enumerate(MY_UNIT_TYPE):
                            feature[feature == unit_type] = j + 1

                    # N x 1 x H x W
                    tmp = torch.LongTensor(feature).cuda().view(1, 1, Params["FeatureSize"], Params["FeatureSize"])
                    one_hots = torch.Tensor(tmp.size(0), scale, tmp.size(2), tmp.size(3)).cuda().zero_()
                    one_hots = one_hots.scatter_(1, tmp.data, 1)
                    categorical = torch.cat((categorical, one_hots[0]))
                else:
                    numerical = torch.cat((numerical, torch.Tensor(feature).cuda().unsqueeze(0)))

            categorical = self.ScrPreproc(categorical.unsqueeze(0))
            numerical = torch.log2(numerical + 1).unsqueeze(0)
            features = torch.cat((categorical, numerical), 1)

        else:
            buffer = torch.Tensor().cuda()
            for i, feature in enumerate(features):
                # N x 1 x H x W
                tmp = torch.LongTensor(feature).cuda().view(1, 1, Params["FeatureSize"], Params["FeatureSize"])
                one_hots = torch.Tensor(tmp.size(0), MINIMAP_FEATURES[i].scale, tmp.size(2), tmp.size(3)).cuda().zero_()
                one_hots = one_hots.scatter_(1, tmp.data, 1)
                buffer = torch.cat((buffer, one_hots[0]))

            features = self.MapPreproc(buffer.unsqueeze(0))

        return features
