from Interfaces.IAgent import IAgent
from Models.TreasureIslandModel_A2C import TImodel_A2C

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as functional


class TIagent(IAgent):

    def __init__(self, feature_size):

        self.model = TImodel_A2C(feature_size)
        self.optim = optim.Adam(self.model.parameters(), lr=0.001)

        self.episode_reward = 0
        self.values, self.entropies, self.logs, self.rewards = [], [], [], []

    def reset(self):
        self.episode_reward = 0

    def action(self, obs):
        logits, value = self.model(obs)

        self.entropies.append(-(functional.softmax(logits, dim=-1) * functional.log_softmax(logits, dim=-1)).sum())
        self.values.append(value)

        policy = functional.softmax(logits[obs.available_actions], dim=-1)
        probabilities = policy.detach().numpy()
        probability = np.random.choice(probabilities, 1, p=probabilities)
        action = np.where(probabilities == probability)[0][0]

        self.logs.append(torch.log(policy[action]))

        return obs.available_actions[action]

    def reward(self, reward):
        self.rewards.append(reward)
        self.episode_reward += reward

    def train(self, obs):
        G = 0

        if not obs.done:
            _, value = self.model(obs)
            G = value.detach().item()

        value_loss = 0
        policy_loss = 0

        for i in reversed(range(len(self.rewards))):
            G = self.rewards[i] + 0.99 * G
            advantage = G - self.values[i]

            value_loss = value_loss + 0.5 * advantage.pow(2)
            policy_loss = policy_loss - (advantage.detach() * self.logs[i] + 0.001 * self.entropies[i])

        #print('value_loss: {}'.format(value_loss.item()))
        #print('policy_loss: {}'.format(policy_loss.item()))

        loss = policy_loss + 0.5 * value_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.values, self.entropies, self.logs, self.rewards = [], [], [], []
