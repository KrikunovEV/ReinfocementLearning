from pysc2.lib import actions as sc2_actions
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.lib import static_data

import matplotlib.pyplot as plt
import numpy as np

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)
np.set_printoptions(threshold=sys.maxsize)

env = sc2_env.SC2Env(
    map_name="BuildMarines",
    step_mul=1,
    visualize=False,
    agent_interface_format=sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=64,
            minimap=64))
)

obs = env.reset()[0]  # num agent
#print(obs.observation)
#obs = env.step(actions=[sc2_actions.FunctionCall(6, [[0]])])[0]
#print(obs.observation)

action_mask = obs.observation["available_actions"]
if 6 not in action_mask:
    obs = env.reset()[0]  # num agent

for i in range(0, 100000000):
    obs = env.step(actions=[sc2_actions.FunctionCall(0, [])])[0]

    if obs.step_type == 2:
        break
'''
#print(obss)
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

print(len(static_data.UNIT_TYPES))

screen_ind = [1, 5, 6, 8, 9, 14, 15]
minimap_ind = [1, 4, 5]
screens_obs_all = obss.observation["feature_screen"]
print(len(screens_obs_all))
print()
print()
print()
minimaps_obs_all = obss.observation["feature_minimap"]
flat = obss.observation["player"]
screens_obs = []
minimap_obs = []

fig=plt.figure(figsize=(16, 9), dpi=80)
for i, screen in enumerate(screens_obs_all):
    fig.add_subplot(3, 6, i+1)
    plt.title(screen_labels[i])
    plt.imshow(np.array(screen), cmap='gray')
    if i in screen_ind:
        screens_obs.append([screen, screen_labels[i]])
#plt.show()
#print(screens_obs[2])

fig=plt.figure(figsize=(16, 9), dpi=80)
for i, minimap in enumerate(minimaps_obs_all):
    fig.add_subplot(2, 4, i+1)
    plt.title(minimap_labels[i])
    plt.imshow(np.array(minimap), cmap='gray')
    if i in minimap_ind:
        minimap_obs.append([minimap, minimap_labels[i]])
#plt.show()

fig=plt.figure(figsize=(16, 9), dpi=80)
for i, obs in enumerate(screens_obs):
    fig.add_subplot(3, 3, i+1)
    plt.title(obs[1])
    plt.imshow(np.array(obs[0]), cmap='gray')
for i, obs in enumerate(minimap_obs):
    fig.add_subplot(3, 3, i+7)
    plt.title(obs[1])
    plt.imshow(np.array(obs[0]), cmap='gray')
#plt.show()

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE = actions.FUNCTIONS.Move_screen.id
'''

'''
env.send_chat_messages("Hey")
for i in range(0,100000000):
    new_obs = env.step(actions=[sc2_actions.FunctionCall(_MOVE, [[0], [0, 25]])])[0]

    flat = obss.observation["player"]
    print(flat)

    if new_obs.reward != 0:
        print('reward: ' + str(new_obs.reward))
    done = new_obs.step_type == 2 # environment.StepType.LAST
    if done:
        break

'''


env.close()