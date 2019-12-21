import torch
import torch.nn as nn


class COMAModel(nn.Module):

    def __init__(self, obs_shape, state_shape, action_shape, n_agents):
        super(COMAModel, self).__init__()

        self.n_agents = n_agents
        self.hidden_states = []

        self.Critic = nn.Sequential(
            nn.Linear(state_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.fc1_0 = nn.Sequential(
            nn.Linear(obs_shape, 512),
            nn.ReLU()
        )
        self.gru_cell_0 = nn.GRUCell(input_size=512, hidden_size=256)
        self.fc2_0 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, action_shape)
        )

        self.fc1_1 = nn.Sequential(
            nn.Linear(obs_shape, 512),
            nn.ReLU()
        )
        self.gru_cell_1 = nn.GRUCell(input_size=512, hidden_size=256)
        self.fc2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, action_shape)
        )

        '''
        self.Policies = []
        for n in range(n_agents):
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
        '''

    def forward(self, obs, state, state_value_only=False):
        V = self.Critic(torch.Tensor(state))
        policies = []

        if not state_value_only:
            o = self.fc1_0(torch.Tensor(obs[0])).unsqueeze(0)
            self.hidden_states[0] = self.gru_cell_0(o, self.hidden_states[0])
            policies.append(self.fc2_0(self.hidden_states[0].squeeze()))

            o = self.fc1_1(torch.Tensor(obs[1])).unsqueeze(0)
            self.hidden_states[1] = self.gru_cell_1(o, self.hidden_states[1])
            policies.append(self.fc2_1(self.hidden_states[1].squeeze()))

        return V, policies

    def reset_hidden_states(self):
        self.hidden_states = [torch.zeros((1, 256)) for n in range(self.n_agents)]
