import torch
import gym
import numpy as np
from visdom import Visdom
from A2CModel import A2CModel


MAX_EPISODES = 2000
T_STEPS = 10
DISCOUNT_FACTOR = 0.99
env = gym.make('CartPole-v0')

vis = Visdom()

model = A2CModel()

CriticOptimizer = torch.optim.Adam(model.CriticParameters(), lr=0.005)#torch.optim.RMSprop(model.CriticParameters(), lr=0.0007, alpha=0.99, eps=0.1)
ActorOptimizer = torch.optim.Adam(model.ActorParameters(), lr=0.001)#torch.optim.RMSprop(model.ActorParameters(), lr=0.00035, alpha=0.99, eps=0.1)
#Optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


reward_layout = dict(title="Rewards", xaxis={'title':'episode'}, yaxis={'title':'reward'})
policy_layout = dict(title="Policy loss", xaxis={'title':'n-step iter'}, yaxis={'title':'loss'})
value_layout = dict(title="Value loss", xaxis={'title':'n-step iter'}, yaxis={'title':'loss'})

REWARDS_DATA = []
VALUELOSS_DATA = []
POLICYLOSS_DATA = []

REWARDS = []
VALUELOSS = []
POLICYLOSS = []


def Train(values, log_probs, entropies, rewards, obs, done):

    G = 0
    if not done:
        _, G = model(torch.Tensor(obs))
        G = G.detach()
    else:
        REWARDS.append(REWARD)

    value_loss = 0
    policy_loss = 0

    for i in reversed(range(len(rewards))):
        G = rewards[i] + DISCOUNT_FACTOR * G
        Advantage = G - values[i]

        value_loss += 0.5 * Advantage.pow(2)
        policy_loss -= Advantage * log_probs[i]  # - 0.01 * entropies[i]

    Loss = policy_loss + value_loss

    ActorOptimizer.zero_grad()
    CriticOptimizer.zero_grad()

    Loss.backward()

    ActorOptimizer.step()
    CriticOptimizer.step()

    return policy_loss, value_loss


for episode in range(MAX_EPISODES):

    obs = env.reset()
    REWARD = 0
    values, log_probs, entropies, rewards = [], [], [], []

    done = False
    step = 0
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
        print(prob)
        log_prob = log_prob[action]

        obs, reward, done, info = env.step(action)
        np.clip(reward, -1, 1)

        entropies.append(-(log_prob * prob).sum())
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        REWARD += reward

        step += 1
        if step % T_STEPS == 0:
            Train(values, log_probs, entropies, rewards, obs, done)
            values, log_probs, entropies, rewards = [], [], [], []


    if done:
        vis.line([REWARD], [episode], update='append', win='reward', name="every")
        vis.update_window_opts('reward', opts={'title': 'Episode rewards', 'xlabel': 'episode', 'ylabel': 'reward'})

    vis.line([policy_loss.item()], [episode], update='append', win='policy')
    vis.line([value_loss.item()], [episode], update='append', win='value')
    vis.update_window_opts('policy', opts={'title': 'policy loss', 'xlabel': 'iter', 'ylabel': 'loss'})
    vis.update_window_opts('value', opts={'title': 'value loss', 'xlabel': 'iter', 'ylabel': 'loss'})


env.close()