from pysc2.lib import actions as sc2_actions
from pysc2.env import sc2_env
from pysc2.lib import actions

import matplotlib.pyplot as plt
import numpy as np

from pysc2.lib import features as sc2_features
if sc2_features.SCREEN_FEATURES[0].type == sc2_features.FeatureType.SCALAR:
    print(True)

'''
env = sc2_env.SC2Env(
    map_name="CollectMineralShards",
    step_mul=8,
    visualize=False,
    agent_interface_format=sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
        screen=64,
        minimap=64))
)

obs = env.reset()[0] # num agent

screens = obs.observation["feature_screen"]
minimaps = obs.observation["feature_minimap"]

import sys
np.set_printoptions(threshold=sys.maxsize)
print(np.array(screens[8]))


obs = env.step(actions=[sc2_actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])])[0]
flat = obs.observation["player"]
print(flat)



env.close()
'''