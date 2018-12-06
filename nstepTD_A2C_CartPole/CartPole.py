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

CriticOptimizer = torch.optim.Adam(model.CriticParameters(), lr=0.01)
<<<<<<< HEAD
ActorOptimizer = torch.optim.Adam(model.ActorParameters(), lr=0.001)
=======
ActorOptimizer = torch.optim.Adam(model.ActorParameters(), lr=0.005)
>>>>>>> e739137383ef983d5fb4c042e35571d43c22f14d

reward_layout = dict(title="Rewards", xaxis={'title':'episode'}, yaxis={'title':'reward'})
policy_layout = dict(title="Policy loss", xaxis={'title':'n-step iter'}, yaxis={'title':'loss'})
value_layout = dict(title="Value loss", xaxis={'title':'n-step iter'}, yaxis={'title':'loss'})

REWARDS_DATA = []
VALUELOSS_DATA = []
POLICYLOSS_DATA = []
EPISODES_DATA = []

REWARDS = []
VALUELOSS = []
POLICYLOSS = []

nstepIter = 0
NSTEPITER = []


def Train(values, log_probs, entropies, rewards, obs, done):

    G = 0
    if not done:
        _, G = model(torch.Tensor(obs))
        G = G.detach()

    value_loss = 0
    policy_loss = 0

    for i in reversed(range(len(rewards))):
        G = rewards[i] + DISCOUNT_FACTOR * G
        Advantage = G - values[i]

        value_loss += 0.5 * Advantage.pow(2)
        policy_loss -= Advantage * log_probs[i] + 0.01 * entropies[i]

    Loss = policy_loss + value_loss

    ActorOptimizer.zero_grad()
    CriticOptimizer.zero_grad()

    Loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

    ActorOptimizer.step()
    CriticOptimizer.step()

    POLICYLOSS.append(policy_loss.item())
    VALUELOSS.append(value_loss.item())

    NSTEPITER.append(nstepIter)
    if nstepIter % 10 == 0:
        VALUELOSS_DATA.append(np.mean(VALUELOSS[len(VALUELOSS) - 10:]))
        POLICYLOSS_DATA.append(np.mean(POLICYLOSS[len(POLICYLOSS) - 10:]))

    trace_value = dict(x=NSTEPITER, y=VALUELOSS, type='custom', mode="lines", name='loss')
    trace_policy = dict(x=NSTEPITER, y=POLICYLOSS, type='custom', mode="lines", name='loss')

    trace2_value = dict(x=NSTEPITER[::10], y=VALUELOSS_DATA,
                  line={'color': 'red', 'width': 4}, type='custom', mode="lines", name='mean loss')
    trace2_policy = dict(x=NSTEPITER[::10], y=POLICYLOSS_DATA,
                        line={'color': 'red', 'width': 4}, type='custom', mode="lines", name='mean loss')

    #vis._send({'data': [trace_value, trace2_value], 'layout': value_layout, 'win': 'valuewin'})
    #vis._send({'data': [trace_policy, trace2_policy], 'layout': policy_layout, 'win': 'policywin'})


for episode in range(MAX_EPISODES):

    #if episode % 100 == 0 and episode != 0:
        #torch.save(model.state_dict(), 'models/episodes_' + str(episode) + '.pt')

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
        log_prob = log_prob[action]

        obs, reward, done, info = env.step(action)
        np.clip(reward, -1, 1)

        entropies.append(-(log_prob * prob).sum())
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        REWARD += reward

        if done:
            REWARDS.append(REWARD)
            break

        step += 1
        if step % T_STEPS == 0:
            nstepIter += 1
            Train(values, log_probs, entropies, rewards, obs, done)
            values, log_probs, entropies, rewards = [], [], [], []

    nstepIter += 1
    Train(values, log_probs, entropies, rewards, obs, done)

    if episode % 10 == 0:
        REWARDS_DATA.append(np.mean(REWARDS[len(REWARDS)-10:]))

    EPISODES_DATA.append(episode)

    trace = dict(x=EPISODES_DATA, y=REWARDS,  type='custom', mode="lines", name='reward')
    trace2 = dict(x=EPISODES_DATA[::10], y=REWARDS_DATA,
                  line={'color': 'red', 'width': 4}, type='custom', mode="lines", name='mean reward')
    vis._send({'data':[trace, trace2], 'layout':reward_layout, 'win':'rewardwin'})


env.close()