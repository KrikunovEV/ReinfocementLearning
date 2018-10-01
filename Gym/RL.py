import gym
from RLutil import *
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v0')

model = torch.nn.Sequential(
    torch.nn.Linear(4, 24),
    torch.nn.ReLU(),
    torch.nn.Linear(24, 24),
    torch.nn.ReLU(),
    torch.nn.Linear(24, 2)
)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

DISCOUNT_FACTOR = 0.9
Rewards = []

MyExperience = Experience()

for episode in range(1, 501):

    obs = env.reset()
    total_reward = 0

    if episode % 250 == 0:
        for group in range(len(optimizer.param_groups)):
            optimizer.param_groups[group]['lr'] *= 0.9

    for i in range(300):

        env.render()

        # get predicted rewards
        predict = model.forward(torch.Tensor(obs))

        # make a decision
        action = torch.argmax(predict).detach().numpy()

        # get useful data
        obs_next, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            reward = -reward

        # record experience and train
        MyExperience.addNewExperience(torch.Tensor(obs), reward, action, torch.Tensor(obs_next), done)
        MyExperience.replayExperience(model=model, loss_fn=loss_fn, optimizer=optimizer, DISCOUNT_FACTOR=DISCOUNT_FACTOR)

        obs = obs_next

        if done:
            Rewards.append(total_reward)
            break

env.close()

plt.plot(np.arange(len(Rewards)), Rewards)
plt.show()