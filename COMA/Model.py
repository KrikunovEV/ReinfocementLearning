import torch
import torch.nn as nn
import torch.nn.functional as functional


class COMAModel(nn.Module):

    def __init__(self, obs_shape, state_shape, action_shape, n_agents):
        super(COMAModel, self).__init__()

        self.n_agents = n_agents

        self.Critic = nn.Sequential(
            nn.Linear(state_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.Policies = []
        for n in n_agents:
            fc1 = nn.Sequential(
                nn.Linear(obs_shape, 512),
                nn.ReLU()
            )
            gru_cell = nn.GRUCell(input_size=512, hidden_size=256)
            fc2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, action_shape)
            )
            self.Policies.append({'fc1': fc1, 'GRU': gru_cell, 'fc2': fc2})

    def forward(self, obs, state):
        V = self.Critic(state)
        policies = []
        for n in self.n_agents:
            o = self.Policies[n]['fc1'](obs[n])
            self.hidden_states[n] = self.Policies[n]['GRU'](o, self.hidden_states[n])
            policies.append(self.Policies[n]['fc2'](self.hidden_states[n]))

        return V, policies

    def reset_hidden_states(self):
        self.hidden_states = [torch.zeros((1, 256)) for n in self.n_agents]
