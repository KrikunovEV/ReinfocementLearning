from Environments.TreasureIsland import TIenv
from Agents.TreasureIslandAgent import TIagent

env = TIenv(frame_rate=120, num_marks=2)
agent = TIagent()

EPISODE_COUNT = 1000
STEP_COUNT = 30

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
        if steps % STEP_COUNT == 0:
            agent.train(obs)

    print(str(episode + 1) + ": " + str(agent.episode_reward) + " rewards")
