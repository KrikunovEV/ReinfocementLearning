from Util import VisdomWrap, Global
from Model import FullyConvModel

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

import numpy as np


class Agent:

    def __init__(self, episode_load, save_path):
        self.save_path = save_path

        self.DataMgr = VisdomWrap()

        self.Model = FullyConvModel().cuda()
        self.Optim = optim.Adam(self.Model.parameters(), lr=Global.Params["LR"])
        if episode_load != 0:
            self._load_agent_state(episode_load)

        self.values, self.entropies, self.spatial_entropies, self.logs, self.rewards = [], [], [], [], []
        self.episode_reward = 0

    def reset(self, episode):
        self.episode_reward = 0
        #for param_group in self.Optim.param_groups:
            #param_group['lr'] = Global.Params["LR"] * (1 - episode / Global.Params["Episodes"])

    def make_decision(self, scr_features, map_features, flat_features, action_mask):
        spatial_q, q, value = self.Model(scr_features, map_features, flat_features)

        entropy = -(functional.log_softmax(q, dim=-1) * functional.softmax(q, dim=-1)).sum()
        spatial_entropy = -(functional.log_softmax(spatial_q, dim=-1) * functional.softmax(spatial_q, dim=-1)).sum()

        available_actions = [i for i, action in enumerate(Global.MY_FUNCTION_TYPE) if action in action_mask]
        q = q[available_actions]

        policy = functional.softmax(q, dim=-1)
        spatial_policy = functional.softmax(spatial_q, dim=-1)

        action_id = self._on_policy_choice(policy)
        probability = policy[action_id]
        action_id = Global.MY_FUNCTION_TYPE[available_actions[action_id]]

        action_args = []
        for arg in Global.FUNCTIONS[action_id].args:
            if len(arg.sizes) == 1:
                action_args.append([0])
            elif len(arg.sizes) > 1:
                spatial_action_id = self._on_policy_choice(spatial_policy)
                probability = probability * spatial_policy[spatial_action_id]
                y = spatial_action_id // Global.Params["FeatureSize"]
                x = spatial_action_id % Global.Params["FeatureSize"]
                action_args.append([x, y])

        self.logs.append(torch.log(probability))
        self.entropies.append(entropy)
        self.spatial_entropies.append(spatial_entropy)
        self.values.append(value)

        return action_id, action_args

    def get_reward(self, reward):
        self.rewards.append(reward)
        self.episode_reward += reward

    def train(self, obs, done):
        G = 0

        if not done:
            scr_features = [obs.observation["feature_screen"][i] for i in Global.scr_indices]
            map_features = [obs.observation["feature_minimap"][i] for i in Global.map_indices]
            flat_features = obs.observation["player"]
            _, _, G = self.Model(scr_features, map_features, flat_features)
            G = G.detach().item()

        #discounted = []
        #for i in reversed(range(len(self.rewards))):
            #G = self.rewards[i] + Global.Params["Discount"] * G
            #discounted.append(G)

        #if np.std(discounted) != 0:
            #discounted = (discounted - np.mean(discounted)) / np.maximum(np.std(discounted), 0.000001)

        value_loss = 0
        policy_loss = 0

        for i in reversed(range(len(self.rewards))):
            #G = self.rewards[i] + Global.Params["Discount"] * G
            G = self.rewards[i] + Global.Params["Discount"] * G
            #advantage = discounted[-i-1] - self.values[i]
            advantage = G - self.values[i]

            value_loss = value_loss + 0.5 * advantage.pow(2)
            policy_loss = policy_loss - (advantage.detach() * self.logs[i] +
                                         Global.Params["Entropy"] * (self.entropies[i] + self.spatial_entropies[i]))

        loss = policy_loss + 0.5 * value_loss

        self.Optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Model.parameters(), Global.Params["GradClip"])
        self.Optim.step()

        self.DataMgr.send_data(value_loss.item(), policy_loss.item(),
                               np.mean([entropy.item() for entropy in self.entropies]),
                               np.mean([entropy.item() for entropy in self.spatial_entropies]), done,
                               self.episode_reward)

        self.values, self.entropies, self.spatial_entropies, self.logs, self.rewards = [], [], [], [], []

    def _on_policy_choice(self, policy, do_normalize=False):
        probabilities = policy.cpu().detach().numpy()
        if do_normalize:
            probabilities /= np.sum(probabilities)
        probability = np.random.choice(probabilities, 1, p=probabilities)
        action_id = np.where(probabilities == probability)[0][0]
        return action_id

    def _load_agent_state(self, episode):
        state = torch.load(self.save_path + str(episode) + '.pt')

        self.Model.load_state_dict(state['model_state'])
        self.Optim.load_state_dict(state['optim_state'])
        VALUELOSS = state['value_loss']
        VALUELOSS_MEAN = state['value_loss_mean']
        POLICYLOSS = state['policy_loss']
        POLICYLOSS_MEAN = state['policy_loss_mean']
        ENTROPY = state['entropy']
        ENTROPY_MEAN = state['entropy_mean']
        SPATIALENTROPY = state['spatial_entropy']
        SPATIALENTROPY_MEAN = state['spatial_entropy_mean']
        REWARDS = state['reward']
        REWARDS_MEAN = state['reward_mean']
        EPISODES = state['episodes']
        NSTEPITER = state['nstepiter']

        self.DataMgr.set_data(VALUELOSS, VALUELOSS_MEAN, POLICYLOSS, POLICYLOSS_MEAN, ENTROPY, ENTROPY_MEAN,
                              SPATIALENTROPY, SPATIALENTROPY_MEAN, REWARDS, REWARDS_MEAN, EPISODES, NSTEPITER)

    def save_agent_state(self, episode):
        VALUELOSS, VALUELOSS_MEAN, POLICYLOSS, POLICYLOSS_MEAN, \
        ENTROPY, ENTROPY_MEAN, SPATIALENTROPY, SPATIALENTROPY_MEAN, \
        REWARDS, REWARDS_MEAN, EPISODES, NSTEPITER = self.DataMgr.get_data()

        state = {
            'model_state': self.Model.state_dict(),
            'optim_state': self.Optim.state_dict(),
            'value_loss': VALUELOSS,
            'value_loss_mean': VALUELOSS_MEAN,
            'policy_loss': POLICYLOSS,
            'policy_loss_mean': POLICYLOSS_MEAN,
            'entropy': ENTROPY,
            'entropy_mean': ENTROPY_MEAN,
            'spatial_entropy': SPATIALENTROPY,
            'spatial_entropy_mean': SPATIALENTROPY_MEAN,
            'reward': REWARDS,
            'reward_mean': REWARDS_MEAN,
            'episodes': EPISODES,
            'nstepiter': NSTEPITER
        }

        torch.save(state, self.save_path + str(episode) + '.pt')


'''
def _track_gradients(self):
    params = list(self.Model.parameters())
    grads = torch.Tensor().cuda()
    for param in params:
        grads = torch.cat((grads, param.grad.detach().view(-1)))
    self.DataMgr.send_grad_data(grads.mean().item(), grads.var().item())
'''
