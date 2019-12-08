from Common.TreasureIsland import TIenv
from .Agent import TIagent

import torch

feature_size = (8, 8)

env = TIenv(frame_rate=0, num_marks=1, feature_size=feature_size)
agent = TIagent(feature_size=feature_size, learning_rate=0.0001)

EPISODE_COUNT = 50000
STEP_COUNT = 40


for episode in range(EPISODE_COUNT):

    obs = env.reset()
    agent.reset()

    steps = 0

    if (episode + 1) % 100 == 0:
        state = {
            'model_state': agent.model.state_dict(),
            'optim_state': agent.optim.state_dict()
        }

        torch.save(state, "models/" + str(episode+1) + '.pt')
        env.save_value_and_policy_map_for_A2C(agent.model, 'images/' + str(episode+1) + '.png')

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
            continue

    print(str(episode + 1) + ": " + str(agent.episode_reward))
