from Agent import Preprocess
from ActorCriticModel import ActorCriticModel
import gym
import numpy as np
import torch


env = gym.make('SpaceInvaders-v0')

MAX_EPISODES = 500
MAX_ACTIONS = 2000

model = ActorCriticModel()
model.load_state_dict(torch.load("trainModels/episodes_1000.pt"))
model.eval()

for episode in range(1, MAX_EPISODES):

    print("Episosde:", episode)

    obs = Preprocess(env.reset())

    for _ in range(MAX_ACTIONS):
        env.render()

        prob, value = model(torch.Tensor(obs[np.newaxis,:,:,:]))
        action = torch.argmax(prob)
        print(action)

        # Make a step
        obs, reward, done, info = env.step(action)
        obs = Preprocess(obs)

        if done:
            break

env.close()