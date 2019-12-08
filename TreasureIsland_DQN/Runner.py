from Common.TreasureIsland import TIenv
from TreasureIsland_DQN.Agent import TIagent

import torch

feature_size = (8, 8)

env = TIenv(frame_rate=0, num_marks=1, feature_size=feature_size)
agent = TIagent(lr=0.001, capacity=50000, batch_size=32, model_update_steps=10)

EPISODE_COUNT = 50000


for episode in range(EPISODE_COUNT):

    obs = env.reset()
    agent.reset()

    steps = 0

    while True:
        env.render()



        action = agent.action(obs)
        obs = env.step(action)
        agent.reward(obs.reward)

        if obs.done:
            agent.train(obs)
            break

        steps += 1

    if (episode + 1) % 100 == 0:
        state = {
            'model_state': agent.policy_model.state_dict(),
            'optim_state': agent.optim.state_dict()
        }

        torch.save(state, "models/" + str(episode+1) + '.pt')

    print(str(episode + 1) + ": " + str(agent.episode_reward))
