import numpy as np
import torch.nn.functional as functional
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch

from smac.env import StarCraft2Env
from COMA.Model import COMAModel


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

def train(n_agents, history, terminated, obs, state, model, optimizer, GAMMA, BETTA, global_step):
    G = 0
    if not terminated:
        G, _ = model(obs, state, state_value_only=True)
        G = G.detach().item()

    value_loss = 0
    policy_loss = 0
    for i in reversed(range(len(history['reward']))):
        G = history['reward'][i] + GAMMA * G

        advantage = G - history['value'][i]

        value_loss = value_loss + 0.5 * advantage.pow(2)
        for n in range(n_agents):
            policy_loss = policy_loss -\
                          advantage.detach() * history['log_policy'][i][n] - BETTA * history['entropy'][i][n]

    loss = policy_loss + 0.5 * value_loss

    mlflow.log_metric('value_loss', 0.5 * value_loss.item(), step=global_step)
    mlflow.log_metric('policy_loss', 0.5 * policy_loss.item(), step=global_step)

    optimizer.zero_grad()
    loss.backward()
    #nn.utils.clip_grad_norm_(self.Model.parameters(), Global.Params["GradClip"])
    optimizer.step()


T = 40
GAMMA = 0.9
BETTA = 0.001
Episodes = 1000
LR = [0.001, 0.0001, 0.0005]

env = StarCraft2Env(map_name="2c_vs_64zg")
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]

print(env_info)

for lr in LR:
    with mlflow.start_run(run_name='learning rate ' + str(lr)):
        model = COMAModel(obs_shape=env_info['obs_shape'], state_shape=env_info['state_shape'],
                          action_shape=n_actions, n_agents=n_agents)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        mlflow.log_param('lr', lr)
        mlflow.log_param('T', T)
        mlflow.log_param('GAMMA', GAMMA)
        mlflow.log_param('BETTA', BETTA)
        mlflow.log_param('Episodes', Episodes)

        global_step = 0

        for e in range(Episodes):
            env.reset()
            terminated = False
            episode_reward = 0
            episode_V_mean = 0
            model.reset_hidden_states()
            history = {'value': [], 'reward': [], 'log_policy': [], 'entropy': []}

            for step in range(1000000):
                obs = env.get_obs()  # for each agent (for decentralized execution)
                state = env.get_state()  # full represent of state (for centralized learning)

                V, policies = model(obs, state)
                episode_V_mean += V.item()

                actions = []
                entropies = []
                log_policies = []
                for agent_id in range(n_agents):
                    avail_actions = env.get_avail_agent_actions(agent_id)
                    avail_actions_ind = np.nonzero(avail_actions)[0]

                    probabilities = functional.softmax(policies[agent_id][avail_actions_ind], dim=-1)
                    probs_detached = probabilities.detach().numpy()
                    probability = np.random.choice(probs_detached, 1, p=probs_detached)

                    action = np.where(probs_detached == probability)[0][0]
                    actions.append(avail_actions_ind[action])

                    log_policies.append(torch.log(probabilities[action]))
                    entropies.append(-(functional.softmax(policies[agent_id], dim=-1) *
                                       functional.log_softmax(policies[agent_id], dim=-1)).sum())

                reward, terminated, _ = env.step(actions)
                episode_reward += reward

                history['value'].append(V)
                history['reward'].append(reward)
                history['log_policy'].append(log_policies)
                history['entropy'].append(entropies)

                global_step += 1

                if terminated:
                    train(n_agents, history, terminated, env.get_obs(), env.get_state(), model, optimizer, GAMMA, BETTA, global_step)
                    mlflow.log_metric('episode_reward', episode_reward, step=e)
                    mlflow.log_metric('episode_V_mean', episode_V_mean / step, step=e)
                    break

                if (step + 1) % T == 0:
                    train(n_agents, history, terminated, env.get_obs(), env.get_state(), model, optimizer, GAMMA, BETTA, global_step)
                    model.reset_hidden_states()
                    history = {'value': [], 'reward': [], 'log_policy': [], 'entropy': []}

            print("Total reward in episode {} = {}".format(e, episode_reward))

        mlflow.pytorch.log_model(model, 'model_' + str(lr))

env.close()
