from Agent import Preprocess
from ActorCriticModel import *
import gym
import numpy as np
import torch


env = gym.make('BreakoutDeterministic-v4')

MAX_EPISODES = 500
MAX_ACTIONS = 50000

model = ActorCriticModel_Breakout()
model.load_state_dict(torch.load("trainModels_Breakout/episodes_1550.pt"))
model.eval()

for episode in range(1, MAX_EPISODES):

    print("Episosde:", episode)

    obs = Preprocess(env.reset())

    cx = torch.zeros(1, 256)
    hx = torch.zeros(1, 256)

    for _ in range(MAX_ACTIONS):
        env.render()

        prob, value, (hx, cx) = model(torch.Tensor(obs[np.newaxis,:,:,:]), (hx, cx))
        prob = torch.nn.functional.softmax(prob, dim=-1)
        prob_np = prob.detach().numpy()[0]
        action = np.random.choice(prob_np, 1, p=prob_np)
        action = np.where(prob_np == action)[0][0]
        #action = torch.argmax(prob)

        # Make a step
        obs, reward, done, info = env.step(action)
        obs = Preprocess(obs)

        if done:
            break

env.close()