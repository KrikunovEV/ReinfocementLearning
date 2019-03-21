import torch
from pysc2.env import sc2_env
from Model import FullyConv

from Util import *


env = sc2_env.SC2Env(
    map_name = "CollectMineralShards",
    step_mul = Hyperparam["GameSteps"],
    visualize = False,
    agent_interface_format = sc2_env.AgentInterfaceFormat(
        feature_dimensions = sc2_env.Dimensions(
        screen = Hyperparam["FeatureSize"],
        minimap = Hyperparam["FeatureSize"]))
)


model = FullyConv()
Optimizer = torch.optim.Adam(model.parameters(), lr=Hyperparam["LR"])


DatamMgr = VisdomWrap()


def Train(values, entropies, spatial_entropies, log_policy, rewards, obs, done):

    G = 0
    if not done:
        screens_obs = []
        for i, screen in enumerate(obs.observation["feature_screen"]):
            if i in screen_ind:
                screens_obs.append(screen)

        minimaps_obs = []
        for i, minimap in enumerate(obs.observation["feature_minimap"]):
            if i in minimap_ind:
                minimaps_obs.append(minimap)

        _, _, G = model(screens_obs, minimaps_obs)
        G = G.detach()

    value_loss = 0
    policy_loss = 0

    for i in reversed(range(len(rewards))):
        G = rewards[i] + Hyperparam["Discount"] * G
        Advantage = G - values[i]

        value_loss += 0.5 * Advantage.pow(2)
        policy_loss -= Advantage.detach() * log_policy[i] + Hyperparam["Entropy"] * entropies[i]

    Loss = policy_loss + 0.5 * value_loss

    Optimizer.zero_grad()

    Loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

    Optimizer.step()

    DatamMgr.SendData(True, value_loss.item(), policy_loss.item(),
                      np.mean([entropy.detach().numpy() for entropy in entropies]),
                      np.mean([entropy.detach().numpy() for entropy in spatial_entropies]), 0)


for episode in range(Hyperparam["Episodes"]):

    if episode % 100 == 0 and episode != 0:
        torch.save(model.state_dict(), 'models/' + str(episode) + '.pt')

    values, entropies, spatial_entropies, log_policy, rewards = [], [], [], [], []
    episode_reward = 0
    step = 0
    done = False
    obs = env.reset()[0]

    while not done:

        screens_obs = []
        for i, screen in enumerate(obs.observation["feature_screen"]):
            if i in screen_ind:
                screens_obs.append(screen)

        minimaps_obs = []
        for i, minimap in enumerate(obs.observation["feature_minimap"]):
            if i in minimap_ind:
                minimaps_obs.append(minimap)

        #flat_obs = obs.observation["player"]

        spatial_logits, logits, value = model(screens_obs, minimaps_obs)

        action_mask = obs.observation["available_actions"]
        logits = logits[action_mask]
        spatial_logits = spatial_logits.flatten()

        probs = torch.nn.functional.softmax(logits, dim=-1)
        spatial_probs = torch.nn.functional.softmax(spatial_logits, dim=-1)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        spatial_log_probs = torch.nn.functional.log_softmax(spatial_logits, dim=-1)

        # action
        probs_np = probs.detach().numpy()
        action = np.random.choice(probs_np, 1, p=probs_np)
        action = np.where(probs_np == action)[0][0]
        prob = probs[action]
        action = action_mask[action]

        probs_np = spatial_probs.detach().numpy()
        args = []#[[0]]
        spatial_log_prob = []
        for arg_type in FUNCTION_TYPES[FUNCTIONS[action].function_type]:
            if len(arg_type) > 1:
                spatial_action = np.random.choice(probs_np, 1, p=probs_np)
                spatial_action = np.where(probs_np == spatial_action)[0][0]
                spatial_log_prob.append(spatial_log_probs[spatial_action])
                prob *= spatial_probs[spatial_action]
                y = action // Hyperparam["FeatureSize"]
                x = action % Hyperparam["FeatureSize"]
                args.append([x, y])
            else:
                #args.append([0])
                print("len less than 2")

        # sc2_actions.FUNCTIONS.select_army.id
        print(action, args)
        obs = env.step(actions=[sc2_actions.FunctionCall(action, args)])[0]

        reward = obs.reward
        #np.clip(reward, -1, 1)

        done = (obs.step_type == 2)

        entropies.append(-(log_probs * probs).sum())
        spatial_entropies.append(-(spatial_log_probs * spatial_probs).sum())
        log_policy.append(torch.log(prob))
        values.append(value)
        rewards.append(reward)

        episode_reward += reward

        if done:
            DatamMgr.SendData(False, 0, 0, 0, 0, episode_reward)
            break

        step += 1
        if step % Hyperparam["Steps"] == 0:
            Train(values, entropies, spatial_entropies, log_policy, rewards, obs, done)
            values, entropies, spatial_entropies, log_policy, rewards = [], [], [], [], []

    Train(values, entropies, spatial_entropies, log_policy, rewards, obs, done)


env.close()