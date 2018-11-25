from Agent import Preprocess
#from ActorCriticModel import ActorCriticModel
from ActorCriticModel_SpaceInvaders import ActorCriticModel
import gym
import numpy as np
import torch


env = gym.make('SpaceInvaders-v0')

MAX_EPISODES = 500
MAX_ACTIONS = 2000

model = ActorCriticModel()
model.load_state_dict(torch.load("trainModels4/episodes_500.pt"))
model.eval()

for episode in range(1, MAX_EPISODES):

    print("Episosde:", episode)

    obs = Preprocess(env.reset())

    for _ in range(MAX_ACTIONS):
        env.render()

        prob, value = model(torch.Tensor(obs[np.newaxis,:,:,:]))
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