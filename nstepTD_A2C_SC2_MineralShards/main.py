from Util import Global
from Agent import Agent

from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions as sc2_actions

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


save_path = 'checkpoints_marines_17/'
episode_load = 1000
if episode_load == 0:
    Global.save(save_path)
else:
    Global.load(save_path)
Global.debug_print()


env = sc2_env.SC2Env(
    map_name="BuildMarines",  # CollectMineralShards
    step_mul=Global.Params["GameSteps"],
    visualize=False,
    agent_interface_format=sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=Global.Params["FeatureSize"],
            minimap=Global.Params["FeatureSize"]))
)


agent = Agent(episode_load, save_path)


for episode in range(episode_load, Global.Params["Episodes"]):

    episode_reward = 0
    step = 0
    done = False
    obs = env.reset()[0]
    agent.reset(episode)

    while 6 in obs.observation["available_actions"]:
        env.close()
        env = sc2_env.SC2Env(
            map_name="BuildMarines",  # CollectMineralShards
            step_mul=Global.Params["GameSteps"],
            visualize=False,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=Global.Params["FeatureSize"],
                    minimap=Global.Params["FeatureSize"]))
        )
        obs = env.reset()[0]
        print('GOT ISSUE')

    while not done:

        scr_features = [obs.observation["feature_screen"][i] for i in Global.scr_indices]
        map_features = [obs.observation["feature_minimap"][i] for i in Global.map_indices]
        flat_features = obs.observation["player"]
        action_mask = obs.observation["available_actions"]

        action_id, action_args = agent.make_decision(scr_features, map_features, flat_features, action_mask)
        obs = env.step(actions=[sc2_actions.FunctionCall(action_id, action_args)])[0]

        agent.get_reward(obs.reward)

        done = (obs.step_type == StepType.LAST)

        if done:
            agent.train(obs, done)  # episode has finished
            break

        step += 1
        if step % Global.Params["Steps"] == 0:
            agent.train(obs, done)  # n-step update

    if (episode + 1) % 50 == 0:
        agent.save_agent_state(episode + 1)

    episode += 1

env.close()
