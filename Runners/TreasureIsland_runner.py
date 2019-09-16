from Environments.TreasureIsland import TIenv
from Agents.TreasureIslandAgent import TIagent

import torch

feature_size = (16, 16)

env = TIenv(frame_rate=120, num_marks=4, feature_size=feature_size)
agent = TIagent(feature_size=feature_size)

EPISODE_COUNT = 2000
STEP_COUNT = 40


for episode in range(EPISODE_COUNT):

    obs = env.reset()
    agent.reset()

    if episode % 100 == 0 and episode != 0:
        state = {
            'model_state': agent.model.state_dict(),
            'optim_state': agent.optim.state_dict()
        }

        torch.save(state, "models/" + str(episode) + '.pt')
        env.save_value_and_policy_map_for_A2C(agent.model, 'images/' + str(episode) + '.png')

    steps = 0

    print(str(episode + 1) + ": ")

    while True:
        env.render()

        action = agent.action(obs)
        obs = env.step(action)
        agent.reward(obs.reward)

        if obs.done:
            agent.train(obs)
            break

        steps += 1
        if steps % STEP_COUNT == 0:
            agent.train(obs)

    print("rewards: " + str(agent.episode_reward))
