import torch
import torch.nn.functional as functional
from pysc2.env import sc2_env
from Model import FullyConv

from Util import *

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

env = sc2_env.SC2Env(
    map_name="BuildMarines",
    step_mul=Params["GameSteps"],
    visualize=False,
    agent_interface_format= sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=Params["FeatureSize"],
            minimap=Params["FeatureSize"]))
)


model = FullyConv()
model.load_state_dict(torch.load("models5/300.pt"))
model.eval()

for episode in range(Params["Episodes"]):

    obs = env.reset()[0]

    while True:

        scr_features = [obs.observation["feature_screen"][i] for i in scr_indices]
        map_features = [obs.observation["feature_minimap"][i] for i in map_indices]
        flat_features = obs.observation["player"]
        action_mask = obs.observation["available_actions"]

        spatial_logits, logits, value = model(scr_features, map_features, flat_features)

        actions_ids = [i for i, action in enumerate(MY_FUNCTION_TYPE) if action in action_mask]
        logits = logits[actions_ids]
        spatial_logits = spatial_logits.flatten()

        probs = functional.softmax(logits, dim=-1)
        spatial_probs = functional.softmax(spatial_logits, dim=-1)

        log_probs = functional.log_softmax(logits, dim=-1)
        spatial_log_probs = functional.log_softmax(spatial_logits, dim=-1)

        probs_detached = probs.cpu().detach().numpy()
        prob = np.random.choice(probs_detached, 1, p=probs_detached)
        action_id = np.where(probs_detached == prob)[0][0]
        action_id = MY_FUNCTION_TYPE[actions_ids[action_id]]  # to get real id

        action_args = []
        for arg in FUNCTIONS[action_id].args:
            if len(arg.sizes) == 1:
                action_args.append([0])
            elif len(arg.sizes) > 1:
                probs_detached = spatial_probs.cpu().detach().numpy()
                spatial_action = np.random.choice(probs_detached, 1, p=probs_detached)
                spatial_action = np.where(probs_detached == spatial_action)[0][0]
                spatial_log_prob = spatial_log_probs[spatial_action]
                y = spatial_action // Params["FeatureSize"]
                x = spatial_action % Params["FeatureSize"]
                action_args.append([x, y])

        obs = env.step(actions=[sc2_actions.FunctionCall(action_id, action_args)])[0]

        if obs.step_type == 2:
            break

env.close()
