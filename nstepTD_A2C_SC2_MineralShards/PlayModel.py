from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import numpy as np

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

def GetXY(mask):
  y, x = mask.nonzero()
  return list(zip(x, y))

class MS_Agent(base_agent.BaseAgent):


  # Initialization
  def setup(self, obs_spec, action_spec):
    super(MS_Agent, self).setup(obs_spec, action_spec)


  # Next episode
  def reset(self):
    super(MS_Agent, self).reset() # episode += 1


  # Act
  def step(self, obs):
    super(MS_Agent, self).step(obs) # steps += 1, reward += obs.reward