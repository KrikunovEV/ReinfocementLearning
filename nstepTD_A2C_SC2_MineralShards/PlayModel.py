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
model.load_state_dict(torch.load("models/1000.pt"))
model.eval()

for episode in range(Hyperparam["Episodes"]):

    obs = env.reset()[0]

    while True:

        screens_obs = []
        for i, screen in enumerate(obs.observation["feature_screen"]):
            if i in screen_ind:
                screens_obs.append(screen)

        minimaps_obs = []
        for i, minimap in enumerate(obs.observation["feature_minimap"]):
            if i in minimap_ind:
                minimaps_obs.append(minimap)

        spatial_logits, logits, value = model(screens_obs, minimaps_obs)

        action_mask = obs.observation["available_actions"]
        #logits = logits[action_mask]
        spatial_logits = spatial_logits.flatten()

        probs = torch.nn.functional.softmax(logits, dim=-1)
        spatial_probs = torch.nn.functional.softmax(spatial_logits, dim=-1)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        spatial_log_probs = torch.nn.functional.log_softmax(spatial_logits, dim=-1)

        # action
        probs_np = probs.detach().numpy()
        action = np.random.choice(probs_np, 1, p=probs_np)
        action = np.where(probs_np == action)[0][0]

        x, y = 0, 0
        if action == 1:
            probs_np = spatial_probs.detach().numpy()
            spatial_action = np.random.choice(probs_np, 1, p=probs_np)
            spatial_action = np.where(probs_np == spatial_action)[0][0]
            spatial_log_prob = spatial_log_probs[spatial_action]
            y = spatial_action // Hyperparam["FeatureSize"]
            x = spatial_action % Hyperparam["FeatureSize"]

        _SELECT_ARMY = FUNCTIONS.select_army.id
        _NO_OP = FUNCTIONS.no_op.id
        _MOVE = FUNCTIONS.Move_screen.id

        # sc2_actions.FUNCTIONS.select_army.id
        if action == 0 and (_NO_OP in action_mask):
            obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])[0]
        elif action == 1 and (_MOVE in action_mask):
            obs = env.step(actions=[sc2_actions.FunctionCall(_MOVE, [[0], [x, y]])])[0]
        else:
            obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [[0]])])[0]


        if obs.step_type == 2:
            break


env.close()