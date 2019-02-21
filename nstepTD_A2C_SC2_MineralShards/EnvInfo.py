from pysc2.lib import actions as sc2_actions
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions

import matplotlib.pyplot as plt
import numpy as np

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

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

# 17
screen_labels = [
    "height_map", "visibility_map", "creep", "power", "player_id",
    "player_relative", "unit_type", "selected", "unit_hit_points",
    "unit_hit_points_ratio", "unit_energy", "unit_energy_ratio", "unit_shields",
    "unit_shields_ratio", "unit_density", "unit_density_aa", "effects"]

# 7
minimap_labels = [
    "height_map", "visibility_map", "creep", "camera", "player_id",
    "player_relative", "selected"]

screen_ind = [1, 5, 8, 9, 14, 15]
minimap_ind = [1, 4, 5]
screens_obs_all = obs.observation["feature_screen"]
minimaps_obs_all = obs.observation["feature_minimap"]
screens_obs = []
minimap_obs = []

fig=plt.figure(figsize=(16, 9), dpi=80)
for i, screen in enumerate(screens_obs_all):
    fig.add_subplot(3, 6, i+1)
    plt.title(screen_labels[i])
    plt.imshow(np.array(screen), cmap='gray')
    if i in screen_ind:
        screens_obs.append([screen, screen_labels[i]])
plt.show()

fig=plt.figure(figsize=(16, 9), dpi=80)
for i, minimap in enumerate(minimaps_obs_all):
    fig.add_subplot(2, 4, i+1)
    plt.title(minimap_labels[i])
    plt.imshow(np.array(minimap), cmap='gray')
    if i in minimap_ind:
        minimap_obs.append([minimap, minimap_labels[i]])
plt.show()

fig=plt.figure(figsize=(16, 9), dpi=80)
for i, obs in enumerate(screens_obs):
    fig.add_subplot(3, 3, i+1)
    plt.title(obs[1])
    plt.imshow(np.array(obs[0]), cmap='gray')
for i, obs in enumerate(minimap_obs):
    fig.add_subplot(3, 3, i+7)
    plt.title(obs[1])
    plt.imshow(np.array(obs[0]), cmap='gray')
plt.show()

env.close()