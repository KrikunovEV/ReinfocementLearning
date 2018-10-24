import os
import torch
from ActorCriticModel import ActorCriticModel
from Agent import Agent
from torch.multiprocessing import Process, Pipe, Lock
from visdom import Visdom


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    vis = Visdom()

    MAX_EPISODES = 100
    MAX_ACTIONS = 1000
    DISCOUNT_FACTOR = 0.99
    STEPS = 32

    GlobalACmodel = ActorCriticModel()
    GlobalACmodel.share_memory()

    optimizer = torch.optim.Adam(GlobalACmodel.parameters(), lr=0.0001)

    lock = Lock()

    num_cpu = 4
    agents = []
    for cpu in range(num_cpu):
        agents.append(Agent(cpu))

    reciver, sender = Pipe()

    agent_threads = []
    for agent in agents:
        thread = Process(target=agent.letsgo, args=(GlobalACmodel, optimizer, lock, sender,
                                                      MAX_EPISODES, MAX_ACTIONS, DISCOUNT_FACTOR, STEPS,))
        thread.start()
        agent_threads.append(thread)

    dones = [False, False, False, False]
    once = True
    episode = 0
    while True:
        (epi, episode_reward, episode_length, episode_mean_value, episode_mean_entropy,
         value_loss, policy_loss, cpu, done) = reciver.recv()
        episode += 1
        dones[cpu] = done

        exit = True
        for d in dones:
            if d == False:
                exit = False
                break

        if exit:
            break

        if episode % 100 == 0:
            with lock:
                torch.save(GlobalACmodel.state_dict(), 'trainModels/episodes_' + str(episode) + '.pt')

        if done:
            continue

        vis.line([episode_reward], [episode], update='append', win='reward')
        vis.line([episode_length], [episode], update='append', win='length')
        vis.line([episode_mean_value], [episode], update='append', win='mean_value')
        vis.line([episode_mean_entropy], [episode], update='append', win='mean_entropy')
        vis.line([value_loss], [episode], update='append', win='value_loss')
        vis.line([policy_loss], [episode], update='append', win='policy_loss')

        if once:
            once = False
            vis.update_window_opts('reward', opts={'title': 'Episode rewards', 'xlabel': 'episode', 'ylabel': 'reward'})
            vis.update_window_opts('length', opts={'title': 'Number of actions', 'xlabel': 'episode', 'ylabel': 'actions'})
            vis.update_window_opts('mean_value', opts={'title': 'Mean value', 'xlabel': 'episode', 'ylabel': 'V'})
            vis.update_window_opts('mean_entropy', opts={'title': 'Mean entropy', 'xlabel': 'episode', 'ylabel': 'entropy'})
            vis.update_window_opts('value_loss', opts={'title': 'Value(critic) loss', 'xlabel': 'episode', 'ylabel': 'loss'})
            vis.update_window_opts('policy_loss', opts={'title': 'Policy(actor) loss', 'xlabel': 'episode', 'ylabel': 'loss'})

    for thread in agent_threads:
        thread.join()

