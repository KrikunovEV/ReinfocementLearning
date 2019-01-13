import os
from ActorCriticModel import *
from Agent import Agent
from torch.multiprocessing import Process, Pipe, Lock
from visdom import Visdom
from SharedOptim import SharedRMSprop
import numpy as np

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    vis = Visdom()
    reward_layout = dict(title="Episode rewards", xaxis={'title': 'episode'}, yaxis={'title': 'reward'})
    policy_layout = dict(title="Policy loss", xaxis={'title': 'n-step iter'}, yaxis={'title': 'loss'})
    value_layout = dict(title="Value loss", xaxis={'title': 'n-step iter'}, yaxis={'title': 'loss'})
    entropy_layout = dict(title="Entropies", xaxis={'title': 'n-step iter'}, yaxis={'title': 'entropy'})

    MAX_EPISODES = 20000
    MAX_ACTIONS = 50000
    DISCOUNT_FACTOR = 0.99
    STEPS = 20

    GlobalModel = ActorCriticModel_Breakout()
    GlobalModel.share_memory()

    Optimizer = None
    #Optimizer = SharedRMSprop(GlobalModel.parameters(), lr=0.0007, eps=0.1, alpha=0.99, momentum=0.0)
    CriticOptimizer = SharedRMSprop(GlobalModel.getCriticParameters(), lr=0.00035, alpha=0.99, eps=0.1, momentum=0.0)
    ActorOptimizer = SharedRMSprop(GlobalModel.getActorParameters(), lr=0.0007, alpha=0.99, eps=0.1, momentum=0.0)
    #Optimizer.share_memory()
    CriticOptimizer.share_memory()
    ActorOptimizer.share_memory()

    lock = Lock()

    num_cpu = 4
    agents = []
    for cpu in range(num_cpu):
        agents.append(Agent(cpu))

    receiver, sender = Pipe()

    agent_threads = []
    for agent in agents:
        thread = Process(target=agent.letsgo, args=(GlobalModel, CriticOptimizer, ActorOptimizer, lock, sender,
                                                      MAX_EPISODES, MAX_ACTIONS, DISCOUNT_FACTOR, STEPS, Optimizer,))
        thread.start()
        agent_threads.append(thread)

    dones = [False for _ in range(num_cpu)]

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

    while True:
        (cpu, is_nstep, value_loss, policy_loss, entropy, reward, complete) = receiver.recv()

        dones[cpu] = complete

        exit = True
        for d in dones:
            if d == False:
                exit = False
                break
        if exit:
            break

        if complete:
            continue

        if is_nstep:

            valueloss_sample.append(value_loss)
            policyloss_sample.append(policy_loss)
            entropy_sample.append(float(entropy))

            if len(valueloss_sample) == 100:
                NSTEPITER.append(len(NSTEPITER) + 1)
                VALUELOSS.append(np.mean(valueloss_sample))
                POLICYLOSS.append(np.mean(policyloss_sample))
                ENTROPY.append(np.mean(entropy_sample))
                valueloss_sample = []
                policyloss_sample = []
                entropy_sample = []

                if len(NSTEPITER) % 10 == 0:
                    VALUELOSS_MEAN.append(np.mean(VALUELOSS[len(VALUELOSS) - 10:]))
                    POLICYLOSS_MEAN.append(np.mean(POLICYLOSS[len(POLICYLOSS) - 10:]))
                    ENTROPY_MEAN.append(np.mean(ENTROPY[len(ENTROPY) - 10:]))

                trace_value = dict(x=NSTEPITER, y=VALUELOSS, type='custom', mode="lines", name='loss')
                trace_policy = dict(x=NSTEPITER, y=POLICYLOSS, type='custom', mode="lines", name='loss')
                trace_entropy = dict(x=NSTEPITER, y=ENTROPY, type='custom', mode="lines", name='entropy')

                trace_value_mean = dict(x=NSTEPITER[::10], y=VALUELOSS_MEAN,
                                    line={'color': 'red', 'width': 3}, type='custom', mode="lines", name='mean loss')
                trace_policy_mean = dict(x=NSTEPITER[::10], y=POLICYLOSS_MEAN,
                                     line={'color': 'red', 'width': 3}, type='custom', mode="lines", name='mean loss')
                trace_entropy_mean = dict(x=NSTEPITER[::10], y=ENTROPY_MEAN,
                                    line={'color': 'red', 'width': 3}, type='custom', mode="lines", name='mean entropy')

                vis._send({'data': [trace_value, trace_value_mean], 'layout': value_layout, 'win': 'valuewin'})
                vis._send({'data': [trace_policy, trace_policy_mean], 'layout': policy_layout, 'win': 'policywin'})
                vis._send({'data': [trace_entropy, trace_entropy_mean], 'layout': entropy_layout, 'win': 'entropywin'})

        else:

            reward_sample.append(reward)
            if(len(reward_sample) == 10):
                EPISODES.append(len(EPISODES) + 1)
                REWARDS.append(np.mean(reward_sample))
                reward_sample = []

                if len(EPISODES) % 10 == 0:
                    REWARDS_MEAN.append(np.mean(REWARDS[len(REWARDS) - 10:]))

                trace_reward = dict(x=EPISODES, y=REWARDS, type='custom', mode="lines", name='reward')
                trace_reward_mean = dict(x=EPISODES[::10], y=REWARDS_MEAN,
                                    line={'color': 'red', 'width': 4}, type='custom', mode="lines", name='mean reward')

                vis._send({'data': [trace_reward, trace_reward_mean], 'layout': reward_layout, 'win': 'rewardwin'})

                if len(EPISODES) % 50 == 0:
                    print("Saved")
                    torch.save(GlobalModel.state_dict(), 'trainModels_Breakout/episodes_' + str(len(EPISODES)) + '.pt')

    for thread in agent_threads:
        thread.join()

