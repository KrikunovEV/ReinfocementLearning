import torch
import gym
import numpy as np
from A2CModel import A2CModel


MAX_EPISODES = 2000
env = gym.make('CartPole-v0')

model = A2CModel()
model.load_state_dict(torch.load("models/episodes_1500.pt"))

for episode in range(MAX_EPISODES):

    obs = env.reset()

    done = False
    while not done:
        env.render()

        # take probs and value
        logit, value = model(torch.Tensor(obs))
        prob = torch.nn.functional.softmax(logit, dim=-1)
        log_prob = torch.nn.functional.log_softmax(logit, dim=-1)

        # take action
        prob_np = prob.detach().numpy()
        action = np.random.choice(prob_np, 1, p=prob_np)
        action = np.where(prob_np == action)[0][0]
        log_prob = log_prob[action]

        obs, reward, done, info = env.step(action)


env.close()