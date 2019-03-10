from pysc2.lib import actions as sc2_actions

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