from pysc2.lib import actions as sc2_actions
from Agent import Agent
from pysc2.env import sc2_env
from Util import Global
from pysc2.env.environment import StepType

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

env = sc2_env.SC2Env(
    map_name="BuildMarines",
    step_mul=4,
    visualize=False,
    agent_interface_format=sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=32,
            minimap=32))
)


agent = Agent(450, "checkpoints_marines_9/")


for episode in range(1000):

    episode_reward = 0
    step = 0
    done = False
    obs = env.reset()[0]
    agent.reset(episode)

    while 6 in obs.observation["available_actions"]:
        env.close()
        env = sc2_env.SC2Env(
            map_name="BuildMarines",
            step_mul=4,
            visualize=False,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=32,
                    minimap=32))
        )
        obs = env.reset()[0]
        print('GOT ISSUE')

    while not done:

        scr_features = [obs.observation["feature_screen"][i] for i in Global.scr_indices]
        map_features = [obs.observation["feature_minimap"][i] for i in Global.map_indices]
        flat_features = obs.observation["player"]
        action_mask = obs.observation["available_actions"]

        action_id, action_args = agent.make_decision(scr_features, map_features, flat_features, action_mask, True)
        obs = env.step(actions=[sc2_actions.FunctionCall(action_id, action_args)])[0]

        if obs.step_type == StepType.LAST:
            print(agent.episode_reward)
            break

env.close()
