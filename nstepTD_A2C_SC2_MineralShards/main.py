from Util import *

from Agent import Agent

from pysc2.env import sc2_env
from pysc2.env.environment import StepType

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


env = sc2_env.SC2Env(
    map_name="BuildMarines",  # CollectMineralShards
    step_mul=Params["GameSteps"],
    visualize=False,
    agent_interface_format=sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=Params["FeatureSize"],
            minimap=Params["FeatureSize"]))
)


agent = Agent()


for episode in range(Params["Episodes"]):

    episode_reward = 0
    step = 0
    done = False
    obs = env.reset()[0]
    agent.reset(episode)

    while not done:

        scr_features = [obs.observation["feature_screen"][i] for i in scr_indices]
        map_features = [obs.observation["feature_minimap"][i] for i in map_indices]
        flat_features = obs.observation["player"]
        action_mask = obs.observation["available_actions"]

        action_id, action_args = agent.make_decision(scr_features, map_features, flat_features, action_mask)
        obs = env.step(actions=[sc2_actions.FunctionCall(action_id, action_args)])[0]

        agent.get_reward(np.clip(obs.reward, -1, 1))

        done = (obs.step_type == StepType.LAST)

        if done:
            agent.train(obs, done)  # episode has finished
            break

        step += 1
        # T = Params["Steps"] + int((Params["MaxSteps"] - Params["Steps"]) * (episode / Params["Episodes"]))
        if step % Params["Steps"] == 0:
            agent.train(obs, done)  # n-step update

    if (episode + 1) % 50 == 0:
        agent.save_agent_state(episode)

env.close()
