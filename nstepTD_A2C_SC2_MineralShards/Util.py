from pysc2.lib import actions as sc2_actions
from pysc2.lib import features as sc2_features
import numpy as np

FUNCTION_TYPES = sc2_actions.FUNCTION_TYPES
FunctionCount = len(FUNCTION_TYPES)

Hyperparam = {
    "Episodes": 100000,
    "Steps": 40,
    "Discount": 0.99,
    "Entropy": 0.001,
    "GameSteps": 8, # 180 APM
    "LR": 0.0001,
    "FeatureSize": 64
}

screen_ind = [1, 5, 8, 9, 14, 15]
minimap_ind = [1, 4, 5]
FeatureScrCount = len(screen_ind)
FeatureMinimapCount = len(minimap_ind)


SCREEN_FEATURES = sc2_features.SCREEN_FEATURES
MINIMAP_FEATURES = sc2_features.MINIMAP_FEATURES
CATEGORICAL = sc2_features.FeatureType.CATEGORICAL
def Preprocess(feature, index, isScreen):

    if isScreen:
        FEATURES = SCREEN_FEATURES
    else:
        FEATURES = MINIMAP_FEATURES

    if FEATURES[index].type == CATEGORICAL:
        pass
    else:
        feature = np.log(feature + 0.00000001)

    return feature