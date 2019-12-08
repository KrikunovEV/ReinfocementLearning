from .ExperienceMemory import ExperienceMemory
from .Model import Model

import torch
import torch.optim as optim
import torch.nn.functional as functional

import numpy as np


class TIagent:

    def __init__(self, discount=0.99, lr=0.001, capacity=50000, batch_size=32, model_update_steps=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_model = Model().to(self.device)
        self.target_model = Model().to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        self.optim = optim.Adam(self.policy_model.parameters(), lr=lr)

        self.experience = ExperienceMemory(capacity, batch_size)

        self.discount = discount
        self.episode_reward = 0
        self.model_update_steps = model_update_steps

    def reset(self):
        self.episode_reward = 0

    def action(self, obs):
        with torch.no_grad():
            return self.policy_model(obs).max(1)[1].view(1, 1)

        policy = functional.softmax(logits[obs.available_actions], dim=-1)
        probabilities = policy.detach().numpy()
        probability = np.random.choice(probabilities, 1, p=probabilities)
        action = np.where(probabilities == probability)[0][0]

        self.logs.append(torch.log(policy[action]))

        return obs.available_actions[action]

    def reward(self, reward):
        self.episode_reward += reward

    def train(self, obs):
        G = 0
        GAMMA = 1

        if not obs.done:
            _, value = self.model(obs)
            G = value.detach().item()

        value_loss = 0
        policy_loss = 0

        for i in reversed(range(len(self.rewards))):
            G = self.rewards[i] + GAMMA * G
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



def getEpsilon(epsilon, threshold):
    if epsilon > threshold:
        return epsilon - 0.000001
    return threshold

def preprocess(img):
    img = img[::2, ::2]
    return np.mean(img, axis=2)[np.newaxis,:,:].astype(np.float32) / 255.0

def getReward(reward):
    return np.sign(reward)

def getQvalues(model, data, gamma):
    obs = torch.Tensor([data[i][0] for i in range(len(data))]).cuda()
    next_obs = torch.Tensor([data[i][1] for i in range(len(data))]).cuda()
    reward = torch.Tensor([data[i][2] for i in range(len(data))]).cuda()
    action = [data[i][3] for i in range(len(data))]
    done = torch.ByteTensor([data[i][4] for i in range(len(data))]).cuda()

    maxQvalues = reward + gamma * torch.max(model.noGradForward(next_obs), 1)[0]
    Qnew = torch.where(done, reward, maxQvalues)

    Qvalues = model.noGradForward(obs)
    for i in range(len(action)):
        Qvalues[i][action[i]] = Qnew[i]

    return Qvalues