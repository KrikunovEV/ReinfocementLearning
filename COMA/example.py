from smac.env import StarCraft2Env
import numpy as np


env = StarCraft2Env(map_name="2c_vs_64zg", step_mul=1)
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]

n_episodes = 100

for e in range(n_episodes):
    env.reset()
    terminated = False
    episode_reward = 0

    steps = 0
    while not terminated:
        obs = env.get_obs()
        state = env.get_state()

        actions = []
        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)

        reward, terminated, _ = env.step(actions)
        episode_reward += reward
        steps += 1

    print("Total reward in episode {} = {}, steps {}".format(e, episode_reward, steps))

env.close()