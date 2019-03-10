import torch
from pysc2.lib import actions as sc2_actions
from pysc2.env import sc2_env
from pysc2.lib import actions
import numpy as np
from visdom import Visdom
from Model import FullyConv

from Util import *

vis = Visdom()

reward_layout = dict(title="Episode rewards", xaxis={'title': 'episode'}, yaxis={'title': 'reward'})
spatial_policy_layout = dict(title="Policy loss", xaxis={'title': 'n-step iter'}, yaxis={'title': 'loss'})
value_layout = dict(title="Value loss", xaxis={'title': 'n-step iter'}, yaxis={'title': 'loss'})
entropy_layout = dict(title="Entropies", xaxis={'title': 'n-step iter'}, yaxis={'title': 'entropy'})

NSTEPITER = []
VALUELOSS = []
VALUELOSS_MEAN = []
valueloss_sample = []
POLICYLOSS = []
POLICYLOSS_MEAN = []
policyloss_sample = []
ENTROPY = []
ENTROPY_MEAN = []
entropy_sample = []

EPISODES = []
REWARDS = []
REWARDS_MEAN = []
reward_sample = []


env = sc2_env.SC2Env(
    map_name = "CollectMineralShards",
    step_mul = Hyperparam["GameSteps"],
    visualize = False,
    agent_interface_format = sc2_env.AgentInterfaceFormat(
        feature_dimensions = sc2_env.Dimensions(
        screen = Hyperparam["FeatureSize"],
        minimap = Hyperparam["FeatureSize"]))
)


model = FullyConv()
Optimizer = torch.optim.Adam(model.parameters(), lr = Hyperparam["LR"])


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
        policy_loss -= Advantage.detach() * log_probs[i] + 0.01 * entropies[i]

    Loss = policy_loss + 0.5 * value_loss

    #ActorOptimizer.zero_grad()
    #CriticOptimizer.zero_grad()
    Optimizer.zero_grad()

    Loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

    #ActorOptimizer.step()
    #CriticOptimizer.step()
    Optimizer.step()

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

    vis._send({'data': [trace_value, trace2_value], 'layout': value_layout, 'win': 'valuewin'})
    vis._send({'data': [trace_policy, trace2_policy], 'layout': policy_layout, 'win': 'policywin'})


for episode in range(Hyperparam["Episodes"]):

    if episode % 100 == 0 and episode != 0:
        torch.save(model.state_dict(), 'models/' + str(episode) + '.pt')

    values, entropies, spatial_entropies, log_probs, spatial_log_probs, rewards = [], [], [], [], [], []
    episode_reward = 0
    done = False
    obs = env.reset()[0]

    while not done:

        screens_obs = []
        for i, screen in enumerate(obs.observation["feature_screen"]):
            if i in screen_ind:
                screens_obs.append(torch.Tensor(screen))

        minimaps_obs = []
        for i, minimap in enumerate(obs.observation["feature_minimap"]):
            if i in minimap_ind:
                minimaps_obs.append(torch.Tensor(minimap))

        flat_obs = obs.observation["player"]

        spatial_logits, logits, value = model(screens_obs, minimaps_obs, flat_obs)

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
            values, entropies, spatial_entropies, log_probs, spatial_log_probs, rewards = [], [], [], [], [], []

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