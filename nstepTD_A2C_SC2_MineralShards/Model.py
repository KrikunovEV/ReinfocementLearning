import torch
from enum import Enum
from pysc2.lib import features as sc2_features
from Util import *

SCREEN_FEATURES = sc2_features.SCREEN_FEATURES
MINIMAP_FEATURES = sc2_features.MINIMAP_FEATURES
CATEGORICAL = sc2_features.FeatureType.CATEGORICAL

class Type(Enum):
    SCREEN = 0
    MINIMAP = 1
    FLAT = 2

class FullyConv(torch.nn.Module):

    def __init__(self):
        super(FullyConv, self).__init__()

        #self.PreprocessConv = torch.nn.Conv2d(Hyperparam["FeatureSize"], 1, 1)

        self.MinimapNet = torch.nn.Sequential(
            torch.nn.Conv2d(FeatureMinimapCount, 16, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU()
        )

        self.ScreenNet = torch.nn.Sequential(
            torch.nn.Conv2d(FeatureScrCount, 16, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU()
        )

        self.SpatialPolicy = torch.nn.Conv2d(32 + 32, 1, 1)

        # features size**2, 32 filters, 2 from minimap and screen nets
        self.LinearNet = torch.nn.Sequential(
            torch.nn.Linear(Hyperparam["FeatureSize"]**2 * 32 * 2, 256),
            torch.nn.ReLU()
        )

        self.Value = torch.nn.Linear(256, 1)
        self.Policy = torch.nn.Linear(256, FunctionCount)


    def forward(self, screens, minimaps):
        screens = torch.Tensor(screens).unsqueeze_(0)
        minimaps = torch.Tensor(minimaps).unsqueeze_(0)

        scr_data = self.ScreenNet(screens).squeeze_(0)
        map_data = self.MinimapNet(minimaps).squeeze_(0)

        conc_data = torch.cat((scr_data, map_data), 0).unsqueeze_(0)

        spatial_logits = self.SpatialPolicy(conc_data)

        nonspatial_data = conc_data.flatten()
        nonspatial_data = self.LinearNet(nonspatial_data)
        logits = self.Policy(nonspatial_data)
        value = self.Value(nonspatial_data)

        return spatial_logits, logits, value


    def Preprocess(self, feature, index, type):

        if type == Type.FLAT:
            pass

        if type == Type.SCREEN:
            FEATURES = SCREEN_FEATURES
        else:
            FEATURES = MINIMAP_FEATURES

        if FEATURES[index].type == CATEGORICAL:
            pass
            '''
            indices = torch.unsqueeze(torch.LongTensor(feature), 2)

            feature = torch.FloatTensor(Hyperparam["FeatureSize"], Hyperparam["FeatureSize"], FEATURES[index].scale).zero_()
            feature.scatter_(2, indices, 1)

            feature = self.PreprocessConv(torch.unsqueeze(feature, 0))[0][0]
            '''
        else:
            feature = torch.log2(torch.Tensor(feature) + 0.00000001)

        return feature