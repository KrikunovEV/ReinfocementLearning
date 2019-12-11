from smac.env import StarCraft2Env
import numpy as np
from .Model import COMAModel
import torch.nn.functional as functional

# action_id meaning       condition
# 0         no_op         unit is dead
# 1         go_up
# 2         go_down
# 3         go_left
# 4         go_right
# 5         attack/heal
# ...
# n         attack n-th enemy
# ...


env = StarCraft2Env(map_name="2c_vs_64zg")
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]

print(env_info)

model = COMAModel(obs_shape=env_info['obs_shape'], state_shape=env_info['state_shape'],
                  action_shape=n_actions, n_agents=n_agents)

T = 40

for e in range(1000):
    env.reset()
    terminated = False
    episode_reward = 0
    model.reset_hidden_states()

    step = 0
    while not terminated:
        obs = env.get_obs()  # for each agent (for decentralized execution)
        state = env.get_state()  # full represent of state (for centralized learning)

        V, policies = model(obs, state)

        actions = []
        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]  # mask for logits, afterwards apply a softmax

            policies[agent_id] = functional.softmax(policies[agent_id][avail_actions_ind], dim=-1)
            probability = np.random.choice(policies[agent_id], 1, p=policies[agent_id])

            action_id = avail_actions_ind[np.where(policies[agent_id] == probability)[0][0]]
            actions.append(action_id)

        reward, terminated, _ = env.step(actions)
        episode_reward += reward

        if terminated:
            # train
            break

        step += 1
        if step % T == 0:
            # train
            model.reset_hidden_states()

    print("Total reward in episode {} = {}".format(e, episode_reward))

env.close()
