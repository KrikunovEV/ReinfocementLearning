from RL import *
from Model import *
import gym
import torch
'''
if load_model_from is not None:
    state = torch.load(load_model_from)
    self.model.load_state_dict(state['model_state'])
    self.optim.load_state_dict(state['optim_state'])
'''

env = gym.make('SpaceInvaders-v0')

MAX_EPISODES = 500
MAX_ACTIONS = 2000

model = Model()
model.load_state_dict(torch.load("trainModels/episodes_100.pt"))
model.eval()

for episode in range(1, MAX_EPISODES):

    print("Episosde:", episode)

    obs = preprocess(env.reset())

    for _ in range(MAX_ACTIONS):
        env.render()

        action = torch.argmax(model(torch.Tensor(obs[np.newaxis,:,:,:]).cuda()))

        # Make a step
        obs, reward, done, info = env.step(action)
        obs = preprocess(obs)

        if done:
            break

env.close()