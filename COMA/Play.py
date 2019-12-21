from smac.env import StarCraft2Env
import numpy as np
from COMA.Model import COMAModel
import torch.nn.functional as functional
import mlflow.pytorch

env = StarCraft2Env(map_name="2c_vs_64zg", step_mul=1)
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]

#model = COMAModel(obs_shape=env_info['obs_shape'], state_shape=env_info['state_shape'],
#                  action_shape=n_actions, n_agents=n_agents)

model = mlflow.pytorch.load_model('mlruns/0/e5d521f4e6504b69a77788e34407def5/artifacts/model_0.0005')
print(model)
#model.load_state_dict(state)

n_episodes = 1000

for e in range(n_episodes):
    env.reset()
    terminated = False
    model.reset_hidden_states()

    while not terminated:
        obs = env.get_obs()
        state = env.get_state()
        V, policies = model(obs, state)
        actions = []

        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]

            probabilities = functional.softmax(policies[agent_id][avail_actions_ind], dim=-1)
            probs_detached = probabilities.detach().numpy()
            probability = np.random.choice(probs_detached, 1, p=probs_detached)

            action = np.where(probs_detached == probability)[0][0]
            actions.append(avail_actions_ind[action])

        reward, terminated, _ = env.step(actions)

env.close()
