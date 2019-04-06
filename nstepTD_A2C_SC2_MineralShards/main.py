import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from pysc2.env import sc2_env
from Model import FullyConv

from Util import *

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


env = sc2_env.SC2Env(
    map_name="CollectMineralShards",
    step_mul=Params["GameSteps"],
    visualize=False,
    agent_interface_format=sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=Params["FeatureSize"],
            minimap=Params["FeatureSize"]))
)

model = FullyConv()
Optimizer = optim.Adam(model.parameters(), lr=Params["LR"])
DataMgr = VisdomWrap()


def train(values, entropies, spatial_entropies, logs, rewards, obs, done):

    G = 0

    if not done:
        scr_features = [obs.observation["feature_screen"][i] for i in scr_indices]
        map_features = [obs.observation["feature_minimap"][i] for i in map_indices]
        flat_features = obs.observation["player"]
        _, _, G = model(scr_features, map_features, flat_features)
        G = G.detach()

    value_loss = 0
    policy_loss = 0

    for i in reversed(range(len(rewards))):
        G = rewards[i] + Params["Discount"] * G
        advantage = G - values[i]

        value_loss = value_loss + 0.5 * advantage.pow(2)
        policy_loss = policy_loss - (advantage.detach() * logs[i] + Params["Entropy"] * entropies[i])

    loss = policy_loss + 0.5 * value_loss

    Optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 40)
    Optimizer.step()

    DataMgr.send_data(True, value_loss.item(), policy_loss.item(),
                      np.mean([entropy.detach().numpy() for entropy in entropies]),
                      np.mean([entropy.detach().numpy() for entropy in spatial_entropies]), 0)


for episode in range(Params["Episodes"]):

    if episode % 100 == 0 and episode != 0:
        torch.save(model.state_dict(), 'models/' + str(episode) + '.pt')

    values, entropies, spatial_entropies, logs, rewards = [], [], [], [], []
    episode_reward = 0
    step = 0
    done = False
    obs = env.reset()[0]

    while not done:

        scr_features = [obs.observation["feature_screen"][i] for i in scr_indices]
        map_features = [obs.observation["feature_minimap"][i] for i in map_indices]
        flat_features = obs.observation["player"]
        action_mask = obs.observation["available_actions"]

        spatial_logits, logits, value = model(scr_features, map_features, flat_features)

        actions_ids = [i for i, action in enumerate(MY_FUNCTION_TYPE) if action in action_mask]
        logits = logits[actions_ids]
        spatial_logits = spatial_logits.flatten()

        probs = functional.softmax(logits, dim=-1)
        spatial_probs = functional.softmax(spatial_logits, dim=-1)

        log_probs = functional.log_softmax(logits, dim=-1)
        spatial_log_probs = functional.log_softmax(spatial_logits, dim=-1)

        probs_detached = probs.detach().numpy()
        prob = np.random.choice(probs_detached, 1, p=probs_detached)
        action_id = np.where(probs_detached == prob)[0][0]
        prob = probs[action_id]  # to get attached tensor
        action_id = MY_FUNCTION_TYPE[actions_ids[action_id]]  # to get real id

        action_args = []
        for arg in FUNCTIONS[action_id].args:
            if len(arg.sizes) == 1:
                action_args.append([0])
            elif len(arg.sizes) > 1:
                probs_detached = spatial_probs.detach().numpy()
                spatial_action = np.random.choice(probs_detached, 1, p=probs_detached)
                spatial_action = np.where(probs_detached == spatial_action)[0][0]
                spatial_log_prob = spatial_log_probs[spatial_action]
                prob = prob * spatial_probs[spatial_action]
                y = spatial_action // Params["FeatureSize"]
                x = spatial_action % Params["FeatureSize"]
                action_args.append([x, y])

        obs = env.step(actions=[sc2_actions.FunctionCall(action_id, action_args)])[0]

        reward = obs.reward
        np.clip(reward, -1, 1)

        done = (obs.step_type == 2)

        entropies.append(-(log_probs * probs).sum())
        spatial_entropies.append(-(spatial_log_probs * spatial_probs).sum())
        logs.append(torch.log(prob))
        values.append(value)
        rewards.append(reward)

        episode_reward += reward

        if done:
            DataMgr.send_data(False, 0, 0, 0, 0, episode_reward)
            break

        step += 1
        if step % Params["Steps"] == 0:
            train(values, entropies, spatial_entropies, logs, rewards, obs, done)
            values, entropies, spatial_entropies, logs, rewards = [], [], [], [], []

        train(values, entropies, spatial_entropies, logs, rewards, obs, done)

env.close()
