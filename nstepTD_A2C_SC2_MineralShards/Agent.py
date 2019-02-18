from pysc2.lib import actions as sc2_actions
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

env = sc2_env.SC2Env(map_name="CollectMineralShards", step_mul=8) # step_mul - num of gamesteps for each agent step

obs = env.reset()

obs[0].observation["screen"][features.SCREEN_FEATURES.pl]


env.close()