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
        self.Optim = optim.RMSprop(self.Model.parameters(), lr=Global.Params["LR"])
        if episode_load != 0:
            self._load_agent_state(episode_load)

        self.values, self.entropies, self.spatial_entropies, self.logs, self.rewards = [], [], [], [], []
        self.episode_reward = 0

    def reset(self):
        self.episode_reward = 0

        # Learning rate linear annulling
        # for param_group in Optimizer.param_groups:
        #    param_group['lr'] = min(Params["LR"] * (1 - episode / Params["Episodes"]), param_group['lr'])

    def make_decision(self, scr_features, map_features, flat_features, action_mask):
        spatial_logits, logits, value = self.Model(scr_features, map_features, flat_features)

        actions_ids = [i for i, action in enumerate(Global.MY_FUNCTION_TYPE) if action in action_mask]
        logits = logits[actions_ids]
        spatial_logits = spatial_logits.flatten()

        probs = functional.softmax(logits, dim=-1)
        spatial_probs = functional.softmax(spatial_logits, dim=-1)

        log_probs = functional.log_softmax(logits, dim=-1)
        spatial_log_probs = functional.log_softmax(spatial_logits, dim=-1)

        probs_detached = probs.cpu().detach().numpy()
        prob = np.random.choice(probs_detached, 1, p=probs_detached)
        action_id = np.where(probs_detached == prob)[0][0]
        prob = probs[action_id]  # to get attached tensor
        action_id = Global.MY_FUNCTION_TYPE[actions_ids[action_id]]  # to get real id

        action_args = []
        for arg in Global.FUNCTIONS[action_id].args:
            if len(arg.sizes) == 1:
                action_args.append([0])
            elif len(arg.sizes) > 1:
                probs_detached = spatial_probs.cpu().detach().numpy()
                spatial_action = np.random.choice(probs_detached, 1, p=probs_detached)
                spatial_action = np.where(probs_detached == spatial_action)[0][0]
                spatial_log_prob = spatial_log_probs[spatial_action]
                prob = prob * spatial_probs[spatial_action]
                y = spatial_action // Global.Params["FeatureSize"]
                x = spatial_action % Global.Params["FeatureSize"]
                action_args.append([x, y])

        self.entropies.append(-(log_probs * probs).sum())
        self.spatial_entropies.append(-(spatial_log_probs * spatial_probs).sum())
        self.logs.append(torch.log(prob))  # don't take from log_probs because of composite of several probs for policy
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
            G = G.detach()

        discounted = []
        for i in reversed(range(len(self.rewards))):
            G = self.rewards[i] + Global.Params["Discount"] * G
            discounted.append(G)

        advantages = (discounted - np.mean(discounted)) / np.std(discounted) - self.values

        value_loss = 0
        policy_loss = 0

        for i in reversed(range(len(self.rewards))):
            value_loss = value_loss + 0.5 * advantages[i].pow(2)
            policy_loss = policy_loss - (advantages[i].detach() * self.logs[i] +
                                         Global.Params["Entropy"] * (self.entropies[i] + self.spatial_entropies[i]))

        loss = policy_loss + 0.5 * value_loss

        self.Optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Model.parameters(), Global.Params["GradClip"])
        self.Optim.step()

        if done:
            self.DataMgr.send_data(False, 0, 0, 0, 0, self.episode_reward)
        else:
            self.DataMgr.send_data(True, value_loss.item(), policy_loss.item(),
                          np.mean([entropy.item() for entropy in self.entropies]),
                          np.mean([entropy.item() for entropy in self.spatial_entropies]), 0)

        self.values, self.entropies, self.spatial_entropies, self.logs, self.rewards = [], [], [], [], []

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
