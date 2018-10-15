from RL import *
from Experience import *
from Model import *
import gym
import random
import torch

env = gym.make('SpaceInvaders-v0')

MAX_EPISODES = 500
MAX_ACTIONS = 2000

BATCH_SIZE = 32
experience = Experience()

EPSILON_THRESHOLD = 0.01
epsilon = 1.0

DISCOUNT_FACTOR = 0.9

model = Model()

loss_fn = torch.nn.SmoothL1Loss().cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)

for episode in range(1, MAX_EPISODES):

    print("Episosde:", episode)
    if episode % 50 == 0:
        print("SAVE MODEL")
        torch.save(model.state_dict(), 'trainModels/episodes_' + str(episode) + '.pt')

    obs = preprocess(env.reset())

    for _ in range(MAX_ACTIONS):
        env.render()

        # E-greedy
        epsilon = getEpsilon(epsilon, EPSILON_THRESHOLD)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(model.noGradForward(torch.Tensor(obs[np.newaxis,:,:,:]).cuda()))


        # Make a step
        obs_next, reward, done, info = env.step(action)
        obs_next = preprocess(obs_next)

        if done:
            reward = -1
        else:
            reward = getReward(reward)


        # Save experience
        experience.addExperience(obs, obs_next, reward, action, done)
        obs = obs_next
        if len(experience.getExperience(BATCH_SIZE)) < 2:
            continue


        # Train
        exp = experience.getExperience(BATCH_SIZE)

        Qvalues = getQvalues(model, exp, DISCOUNT_FACTOR)

        data = torch.Tensor([exp[i][0] for i in range(len(exp))]).cuda()
        data = model(data)

        loss = loss_fn(data, Qvalues)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break

env.close()