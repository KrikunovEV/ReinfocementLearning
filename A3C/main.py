#import os
#os.environ["OMP_NUM_THREADS"] = "1"

import threading
from multiprocessing import cpu_count
from ActorCriticModel import ActorCriticModel
from Agent import Agent

MAX_EPISODES = 100
MAX_ACTIONS = 1000
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 32

GlobalACmodel = ActorCriticModel('global')

num_agents = cpu_count()
agents = []
for scope in range(num_agents):
    agents.append(Agent(GlobalACmodel, scope))

agent_threads = []
for agent in agents:
    agent_go = lambda: agent.start(MAX_EPISODES, MAX_ACTIONS, DISCOUNT_FACTOR, BATCH_SIZE)
    thread = threading.Thread(target=(agent_go))
    #thread.daemon = True
    thread.start()
    sleep(0.5)
    agent_threads.append(thread)

while any(thread.is_alive() for thread in agent_threads):
    pass