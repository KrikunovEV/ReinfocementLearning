from pysc2.lib import actions as sc2_actions
from pysc2.lib import features as sc2_features
import numpy as np
from visdom import Visdom
from enum import Enum

FUNCTION_TYPES = sc2_actions.FUNCTION_TYPES
FUNCTIONS = sc2_actions.FUNCTIONS


MY_FUNCTION_TYPE = [
    #0,      # no op
    2,      # select point
    #3,      # select rect
    #5,      # select unit
    #7,      # select army (F2)
    6,      # select idle worker (F1)
    11,     # build queue
    42,     # build barracks
    91,     # build supply depot
    268,    # gathering SCV
    331,    # move
    477,    # train Marine
    490     # train SCV
]
FunctionCount = len(MY_FUNCTION_TYPE)


MY_UNIT_TYPE = [
    18,   # command center
    19,   # supply depot
    21,   # barracks
    45,   # SCV
    48,   # marine
    341   # mineral field
    #1680 # FAKE MINERAL
]


Params = {
    "Episodes": 1500,
    "Steps": 500,
    "Discount": 0.99,
    "GradClip": 40,
    "Entropy": 0.001,  # 0.001
    "GameSteps": 8,  # 180 APM
    "LR": 0.00025,
    "FeatureSize": 32,
    "ScrPreprocNum": 5 + len(MY_UNIT_TYPE)+1 + 2,
    "MapPreprocNum": 5 + 2
}


class Type(Enum):
    SCREEN = 0
    MINIMAP = 1
    FLAT = 2


scr_indices = [5, 6, 7, 14, 15]
map_indices = [5, 6]
FeatureScrCount = len(scr_indices)
FeatureMinimapCount = len(map_indices)


SCREEN_FEATURES = [sc2_features.SCREEN_FEATURES[i] for i in scr_indices]
MINIMAP_FEATURES = [sc2_features.MINIMAP_FEATURES[i] for i in map_indices]
CATEGORICAL = sc2_features.FeatureType.CATEGORICAL


class VisdomWrap:

    def __init__(self):
        self.vis = Visdom()

        self.reward_layout = dict(title="Episode rewards", xaxis={'title': 'episode'}, yaxis={'title': 'reward'})
        self.policy_layout = dict(title="Policy loss", xaxis={'title': 'n-step iter'}, yaxis={'title': 'loss'})
        self.value_layout = dict(title="Value loss", xaxis={'title': 'n-step iter'}, yaxis={'title': 'loss'})
        self.entropy_layout = dict(title="Entropies", xaxis={'title': 'n-step iter'}, yaxis={'title': 'entropy'})
        self.spatial_entropy_layout = dict(title="Spatial entropies", xaxis={'title': 'n-step iter'},
                                           yaxis={'title': ' spatial entropy'})

        self.gradmean_layout = dict(title="Grad means", xaxis={'title': 'backward iter'}, yaxis={'title': 'mean'})
        self.gradvar_layout = dict(title="Grad vars", xaxis={'title': 'backward iter'}, yaxis={'title': 'variance'})
        self.GRADMEANS = []
        self.GRADVARS = []
        self.GRADMEANS_MEAN = []
        self.GRADVARS_MEAN = []
        self.gradmeans_sample = []
        self.gradvars_sample = []
        self.BACKWARDS = []

        self.NSTEPITER = []
        self.VALUELOSS = []
        self.VALUELOSS_MEAN = []
        self.valueloss_sample = []
        self.POLICYLOSS = []
        self.POLICYLOSS_MEAN = []
        self.policyloss_sample = []
        self.ENTROPY = []
        self.ENTROPY_MEAN = []
        self.entropy_sample = []
        self.SPATIALENTROPY = []
        self.SPATIALENTROPY_MEAN = []
        self.spatial_entropy_sample = []

        self.EPISODES = []
        self.REWARDS = []
        self.REWARDS_MEAN = []
        self.reward_sample = []

    def send_data(self, is_nstep, value_loss, policy_loss, entropy, spatial_entropy, reward):

        if is_nstep:

            self.valueloss_sample.append(value_loss)
            self.policyloss_sample.append(policy_loss)
            self.entropy_sample.append(float(entropy))
            self.spatial_entropy_sample.append(float(spatial_entropy))

            if len(self.valueloss_sample) == 5:
                self.NSTEPITER.append(len(self.NSTEPITER) + 1)
                self.VALUELOSS.append(np.mean(self.valueloss_sample))
                self.POLICYLOSS.append(np.mean(self.policyloss_sample))
                self.ENTROPY.append(np.mean(self.entropy_sample))
                self.SPATIALENTROPY.append(np.mean(self.spatial_entropy_sample))

                self.valueloss_sample = []
                self.policyloss_sample = []
                self.entropy_sample = []
                self.spatial_entropy_sample = []

                if len(self.NSTEPITER) % 10 == 0:
                    self.VALUELOSS_MEAN.append(np.mean(self.VALUELOSS[len(self.VALUELOSS) - 10:]))
                    self.POLICYLOSS_MEAN.append(np.mean(self.POLICYLOSS[len(self.POLICYLOSS) - 10:]))
                    self.ENTROPY_MEAN.append(np.mean(self.ENTROPY[len(self.ENTROPY) - 10:]))
                    self.SPATIALENTROPY_MEAN.append(np.mean(self.SPATIALENTROPY[len(self.SPATIALENTROPY) - 10:]))

                trace_value = dict(x=self.NSTEPITER, y=self.VALUELOSS, type='custom', mode="lines", name='loss')
                trace_policy = dict(x=self.NSTEPITER, y=self.POLICYLOSS, type='custom', mode="lines", name='loss')
                trace_entropy = dict(x=self.NSTEPITER, y=self.ENTROPY, type='custom', mode="lines", name='entropy')
                trace_spatial_entropy = dict(x=self.NSTEPITER, y=self.SPATIALENTROPY, type='custom', mode="lines",
                                             name='spatial entropy')

                trace_value_mean = dict(x=self.NSTEPITER[::10], y=self.VALUELOSS_MEAN,
                                        line={'color': 'red', 'width': 3}, type='custom', mode="lines",
                                        name='mean loss')
                trace_policy_mean = dict(x=self.NSTEPITER[::10], y=self.POLICYLOSS_MEAN,
                                         line={'color': 'red', 'width': 3}, type='custom', mode="lines",
                                         name='mean loss')
                trace_entropy_mean = dict(x=self.NSTEPITER[::10], y=self.ENTROPY_MEAN,
                                          line={'color': 'red', 'width': 3}, type='custom', mode="lines",
                                          name='mean entropy')
                trace_spatial_entropy_mean = dict(x=self.NSTEPITER[::10], y=self.SPATIALENTROPY_MEAN,
                                                  line={'color': 'red', 'width': 3}, type='custom', mode="lines",
                                                  name='mean spatial entropy')

                self.vis._send(
                    {'data': [trace_value, trace_value_mean], 'layout': self.value_layout, 'win': 'valuewin'})
                self.vis._send(
                    {'data': [trace_policy, trace_policy_mean], 'layout': self.policy_layout, 'win': 'policywin'})
                self.vis._send(
                    {'data': [trace_entropy, trace_entropy_mean], 'layout': self.entropy_layout, 'win': 'entropywin'})
                self.vis._send(
                    {'data': [trace_spatial_entropy, trace_spatial_entropy_mean], 'layout': self.spatial_entropy_layout,
                     'win': 'spatial_entropywin'})

        else:

            self.reward_sample.append(reward)
            if len(self.reward_sample) == 1:
                self.EPISODES.append(len(self.EPISODES) + 1)
                self.REWARDS.append(np.mean(self.reward_sample))
                self.reward_sample = []

                if len(self.EPISODES) % 10 == 0:
                    self.REWARDS_MEAN.append(np.mean(self.REWARDS[len(self.REWARDS) - 10:]))

                trace_reward = dict(x=self.EPISODES, y=self.REWARDS, type='custom', mode="lines", name='reward')
                trace_reward_mean = dict(x=self.EPISODES[::10], y=self.REWARDS_MEAN,
                                         line={'color': 'red', 'width': 4}, type='custom', mode="lines",
                                         name='mean reward')

                self.vis._send(
                    {'data': [trace_reward, trace_reward_mean], 'layout': self.reward_layout, 'win': 'rewardwin'})

    def send_grad_data(self, mean, var):
        self.gradmeans_sample.append(mean)
        self.gradvars_sample.append(var)
        if len(self.gradmeans_sample) == 5:
            self.BACKWARDS.append(len(self.BACKWARDS) + 1)
            self.GRADMEANS.append(np.mean(self.gradmeans_sample))
            self.GRADVARS.append(np.mean(self.gradvars_sample))
            self.gradmeans_sample = []
            self.gradvars_sample = []

            if len(self.BACKWARDS) % 10 == 0:
                self.GRADMEANS_MEAN.append(np.mean(self.GRADMEANS[len(self.GRADMEANS) - 10:]))
                self.GRADVARS_MEAN.append(np.mean(self.GRADVARS[len(self.GRADVARS) - 10:]))

            trace_gradmean = dict(x=self.BACKWARDS, y=self.GRADMEANS, type='custom', mode="lines", name='mean')
            trace_gradmean_mean = dict(x=self.BACKWARDS[::10], y=self.GRADMEANS_MEAN,
                                     line={'color': 'red', 'width': 4}, type='custom', mode="lines",
                                     name='mean mean')
            trace_gradvar = dict(x=self.BACKWARDS, y=self.GRADVARS, type='custom', mode="lines", name='variance')
            trace_gradvar_mean = dict(x=self.BACKWARDS[::10], y=self.GRADVARS_MEAN,
                                       line={'color': 'red', 'width': 4}, type='custom', mode="lines",
                                       name='mean variance')

            self.vis._send(
                {'data': [trace_gradmean, trace_gradmean_mean], 'layout': self.gradmean_layout, 'win': 'meandwin'})

            self.vis._send(
                {'data': [trace_gradvar, trace_gradvar_mean], 'layout': self.gradvar_layout, 'win': 'varwin'})
