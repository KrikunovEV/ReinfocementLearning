from pysc2.lib import actions as sc2_actions
from pysc2.lib import features as sc2_features

from numpy import mean
from visdom import Visdom
from enum import Enum

import torch


class Global:

    MY_FUNCTION_TYPE = [
        #0,      # no op
        2,      # select point
        #3,      # select rect
        #7,      # select army (F2)
        6,      # select idle worker (F1)
        11,     # build queue
        42,     # build barracks
        91,     # build supply depot
        264,    # gathering SCV
        #331,    # move
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
    ]
    UnitCount = len(MY_UNIT_TYPE)

    Params = {
        "Episodes":  1000,
        "Steps":     500,  # 560 steps = 1/3 of 14min
        "MaxSteps":  1800,  # 15 min (600 steps = 5 min) since GameSteps is 8
        "Discount":  0.99,
        "GradClip":  40,
        "Entropy":   0.001,  # 0.001
        "GameSteps": 8,  # 180 APM
        "LR":        0.0002,
        "FeatureSize":   32,
        "ScrPreprocNum": 5 + UnitCount+1 + 2,
        "MapPreprocNum": 5 + 2
    }

    scr_indices = [5, 6, 7, 14, 15]
    map_indices = [5, 6]
    FeatureScrFlatCount = 2
    FeatureScrCount = len(scr_indices) - FeatureScrFlatCount
    FeatureMinimapCount = len(map_indices)

    class Type(Enum):
        SCREEN = 0
        MINIMAP = 1
        FLAT = 2

    FUNCTIONS = sc2_actions.FUNCTIONS
    SCREEN_FEATURES = [sc2_features.SCREEN_FEATURES[i] for i in scr_indices]
    MINIMAP_FEATURES = [sc2_features.MINIMAP_FEATURES[i] for i in map_indices]
    CATEGORICAL = sc2_features.FeatureType.CATEGORICAL

    @staticmethod
    def save(save_path):
        state = {
            'func_type': Global.MY_FUNCTION_TYPE,
            'unit_type': Global.MY_UNIT_TYPE,
            'params': Global.Params,
            'scr_flat_count': Global.FeatureScrFlatCount,
            'scr_ind': Global.scr_indices,
            'map_ind': Global.map_indices,
        }

        torch.save(state, save_path + 'state.pt')

    @staticmethod
    def load(save_path):
        state = torch.load(save_path + 'state.pt')

        Global.MY_FUNCTION_TYPE = state['func_type']
        Global.FunctionCount = len(Global.MY_FUNCTION_TYPE)

        Global.MY_UNIT_TYPE = state['unit_type']
        Global.UnitCount = len(Global.MY_UNIT_TYPE)

        Global.Params = state['params']

        Global.FeatureScrFlatCount = state['scr_flat_count']

        Global.scr_indices = state['scr_ind']
        Global.FeatureScrCount = len(Global.scr_indices) - Global.FeatureScrFlatCount
        Global.SCREEN_FEATURES = [sc2_features.SCREEN_FEATURES[i] for i in Global.scr_indices]

        Global.map_indices = state['map_ind']
        Global.FeatureMinimapCount = len(Global.map_indices)
        Global.MINIMAP_FEATURES = [sc2_features.MINIMAP_FEATURES[i] for i in Global.map_indices]

    @staticmethod
    def debug_print():
        print('Functions ids:')
        print(Global.MY_FUNCTION_TYPE)
        print('Amount:', Global.FunctionCount)
        print()
        print('Units ids:')
        print(Global.MY_UNIT_TYPE)
        print('Amount:', Global.UnitCount)
        print()
        print('Params:')
        print(Global.Params)
        print()
        print('Screen features ids:')
        print(Global.scr_indices)
        print('Scalar amount:', Global.FeatureScrFlatCount, '; Categorical amount:', Global.FeatureScrCount)
        print()
        print('Minimap features ids:')
        print(Global.map_indices)
        print('Categorical amount:', Global.FeatureMinimapCount)


class VisdomWrap:

    def __init__(self):
        self.vis = Visdom()

        self.reward_layout = dict(title="Episode rewards", xaxis={'title': 'episode'}, yaxis={'title': 'reward'})
        self.policy_layout = dict(title="Policy loss", xaxis={'title': 'iter'}, yaxis={'title': 'loss'})
        self.value_layout = dict(title="Value loss", xaxis={'title': 'iter'}, yaxis={'title': 'loss'})
        self.entropy_layout = dict(title="Entropies", xaxis={'title': 'iter'}, yaxis={'title': 'entropy'})
        self.spatial_entropy_layout = dict(title="Spatial entropies", xaxis={'title': 'iter'},
                                           yaxis={'title': 'spatial entropy'})

        self.ITER = []
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

    def get_data(self):
        return self.VALUELOSS, self.VALUELOSS_MEAN, self.POLICYLOSS, self.POLICYLOSS_MEAN,\
               self.ENTROPY, self.ENTROPY_MEAN, self.SPATIALENTROPY, self.SPATIALENTROPY_MEAN,\
               self.REWARDS, self.REWARDS_MEAN, self.EPISODES, self.ITER

    def set_data(self, VALUELOSS, VALUELOSS_MEAN, POLICYLOSS, POLICYLOSS_MEAN, ENTROPY, ENTROPY_MEAN,
                 SPATIALENTROPY, SPATIALENTROPY_MEAN, REWARDS, REWARDS_MEAN, EPISODES, ITER):
        self.VALUELOSS = VALUELOSS
        self.VALUELOSS_MEAN = VALUELOSS_MEAN
        self.POLICYLOSS = POLICYLOSS
        self.POLICYLOSS_MEAN = POLICYLOSS_MEAN
        self.ENTROPY = ENTROPY
        self.ENTROPY_MEAN = ENTROPY_MEAN
        self.SPATIALENTROPY = SPATIALENTROPY
        self.SPATIALENTROPY_MEAN = SPATIALENTROPY_MEAN
        self.REWARDS = REWARDS
        self.REWARDS_MEAN = REWARDS_MEAN
        self.EPISODES = EPISODES
        self.ITER = ITER

    def send_data(self, value_loss, policy_loss, entropy, spatial_entropy, done, reward):

        self.valueloss_sample.append(value_loss)
        self.policyloss_sample.append(policy_loss)
        self.entropy_sample.append(float(entropy))
        self.spatial_entropy_sample.append(float(spatial_entropy))

        if len(self.valueloss_sample) == 10:
            self.ITER.append(len(self.ITER) * 10)
            self.VALUELOSS.append(mean(self.valueloss_sample))
            self.POLICYLOSS.append(mean(self.policyloss_sample))
            self.ENTROPY.append(mean(self.entropy_sample))
            self.SPATIALENTROPY.append(mean(self.spatial_entropy_sample))

            self.valueloss_sample = []
            self.policyloss_sample = []
            self.entropy_sample = []
            self.spatial_entropy_sample = []

            if len(self.ITER) % 10 == 0:
                self.VALUELOSS_MEAN.append(mean(self.VALUELOSS[len(self.VALUELOSS) - 10:]))
                self.POLICYLOSS_MEAN.append(mean(self.POLICYLOSS[len(self.POLICYLOSS) - 10:]))
                self.ENTROPY_MEAN.append(mean(self.ENTROPY[len(self.ENTROPY) - 10:]))
                self.SPATIALENTROPY_MEAN.append(mean(self.SPATIALENTROPY[len(self.SPATIALENTROPY) - 10:]))

            trace_value = dict(x=self.ITER, y=self.VALUELOSS, type='custom', mode="lines", name='loss')
            trace_policy = dict(x=self.ITER, y=self.POLICYLOSS, type='custom', mode="lines", name='loss')
            trace_entropy = dict(x=self.ITER, y=self.ENTROPY, type='custom', mode="lines", name='entropy')
            trace_spatial_entropy = dict(x=self.ITER, y=self.SPATIALENTROPY, type='custom', mode="lines",
                                         name='spatial entropy')

            trace_value_mean = dict(x=[x+45 for x in self.ITER[::10]], y=self.VALUELOSS_MEAN,
                                    line={'color': 'red', 'width': 3}, type='custom', mode="lines",
                                    name='mean loss')
            trace_policy_mean = dict(x=[x+45 for x in self.ITER[::10]], y=self.POLICYLOSS_MEAN,
                                     line={'color': 'red', 'width': 3}, type='custom', mode="lines",
                                     name='mean loss')
            trace_entropy_mean = dict(x=[x+45 for x in self.ITER[::10]], y=self.ENTROPY_MEAN,
                                      line={'color': 'red', 'width': 3}, type='custom', mode="lines",
                                      name='mean entropy')
            trace_spatial_entropy_mean = dict(x=[x+45 for x in self.ITER[::10]], y=self.SPATIALENTROPY_MEAN,
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

        if done:

            self.reward_sample.append(reward)
            if len(self.reward_sample) == 1:
                self.EPISODES.append(len(self.EPISODES))
                self.REWARDS.append(mean(self.reward_sample))
                self.reward_sample = []

                if len(self.EPISODES) % 10 == 0:
                    self.REWARDS_MEAN.append(mean(self.REWARDS[len(self.REWARDS) - 10:]))

                trace_reward = dict(x=self.EPISODES, y=self.REWARDS, type='custom', mode="lines", name='reward')
                trace_reward_mean = dict(x=[x+4.5 for x in self.EPISODES[::10]], y=self.REWARDS_MEAN,
                                         line={'color': 'red', 'width': 4}, type='custom', mode="lines",
                                         name='mean reward')

                self.vis._send(
                    {'data': [trace_reward, trace_reward_mean], 'layout': self.reward_layout, 'win': 'rewardwin'})


'''
FUNCTIONS:
0,      # no op
2,      # select point
3,      # select rect
5,      # select unit
7,      # select army (F2)
6,      # select idle worker (F1)
11,     # build queue
42,     # build barracks
91,     # build supply depot
268,    # gathering SCV
331,    # move
477,    # train Marine
490     # train SCV


UNITS:
18,   # command center
19,   # supply depot
21,   # barracks
45,   # SCV
48,   # marine
341   # mineral field
1680  # FAKE MINERAL


'''
