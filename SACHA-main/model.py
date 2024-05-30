import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# config
import yaml
config = yaml.safe_load(open("./configs/config_base.yaml", 'r'))
policy_input_shape = (config['K_obs'], 3, config['fov_size'][0], config['fov_size'][1])
policy_output_shape = config['action_dim']
action_dim = config['action_dim']
hidden_channels = config['hidden_channels']
hidden_dim = config['hidden_dim']
fov_size = tuple(config['fov_size'])
obs_r = (int(np.floor(fov_size[0]/2)), int(np.floor(fov_size[1]/2)))
num_heads = config['num_heads']
K_obs = config['K_obs']

class AttentionPolicy(nn.Module):
    def _init_(self, communication, input_shape=policy_input_shape, output_shape=policy_output_shape,
                 hidden_channels=hidden_channels, hidden_dim=hidden_dim, num_heads=num_heads):
        super()._init_()
        self.communication = communication
        self.input_shape = policy_input_shape
        self.output_shape = policy_output_shape
        self.hidden_channels = hidden_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=16, kernel_size=1),
            nn.ReLU(True),
            nn.Flatten(0),
        )
        self.memory_encoder = nn.GRUCell(16 * self.input_shape[2] * self.input_shape[3], self.hidden_dim) #capture temporal dependencies in sequential data.
        self.value_decoder = nn.Linear(self.hidden_dim, 1) #decodes the hidden state obtained from the memory unit into a scalar value, representing the estimated value function.
        self.advantage_decoder = nn.Linear(self.hidden_dim, self.output_shape) #decodes the hidden state into a vector of action advantages, representing the advantage values associated with different actions.
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_adj_from_states(self, state):
        num_agents = len(state)
        adj = np.zeros((num_agents, num_agents), dtype=float)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                x_i, y_i = state[i][0], state[i][1]
                x_j, y_j = state[j][0], state[j][1]
                if abs(x_i - x_j) <= obs_r[0] and abs(y_i - y_j) <= obs_r[1]:
                    adj[i][j] = 1.0
                    adj[j][i] = 1.0
        return adj

    def forward(self, obs, hidden, state):
        num_agents = len(state)
        next_hidden, latent = [], []
        obs, hidden = obs.float(), hidden.float()
        for i in range(num_agents):
            o_i = [self.obs_encoder(obs[i, k,:]) for k in range(K_obs)]
            o_i = [self.memory_encoder(o, h) for o, h in zip(o_i, hidden)]
            o_i = torch.stack(o_i)
            next_hidden.append(torch.sum(o_i, 0))
            latent.append(torch.sum(o_i, 0))
        V_n = [self.value_decoder(x) for x in latent]
        A_n = [self.advantage_decoder(x) for x in latent]
        Q_n = [V + A - A.mean(0, keepdim=True) for V, A in zip(V_n, A_n)]
        log_pi = [F.log_softmax(Q, dim=0) for Q in Q_n] # transforms the raw outputs of the neural network into a vector of probabilities
        action = [torch.argmax(Q, dim=0) for Q in Q_n]
        return action, torch.stack(next_hidden), log_pi
    
    def group_agents(self, state):
    # Function to group agents based on their heuristics
        sorted_agents = torch.argsort(self.heuristics)
        groups = []
        labels = {}  # Dictionary to store labels for each agent
        current_group = [sorted_agents[0]]
        for i in range(1, len(sorted_agents)):
            # Check if the heuristic of the current agent is within 10 of the heuristic of the previous agent
            if abs(self.heuristics[sorted_agents[i]] - self.heuristics[current_group[-1]]) <= 10:
                # If yes, add the current agent to the current group and label it as 'paired'
                current_group.append(sorted_agents[i])
                labels[sorted_agents[i]] = 'paired'
            else:
                # If not, start a new group with the current agent and label it as 'single'
                groups.append(current_group)
                labels[sorted_agents[i]] = 'single'
                current_group = [sorted_agents[i]]
        # Add the last group
        groups.append(current_group)
        labels[sorted_agents[-1]] = 'single'
        single_count = sum(1 for label in labels.values() if label == 'single')
        paired_count = sum(1 for label in labels.values() if label == 'paired')
    
        return groups,single_count, paired_count



class AttentionCritic(nn.Module):
    def _init_(self, action_dim=policy_output_shape, hidden_dim=hidden_dim,
                 hidden_channels=hidden_channels, num_heads=num_heads):
        super()._init_()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim    
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.fov_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=16, kernel_size=1),
            nn.ReLU(True),
            nn.Flatten(0),
        )
        self.obs_encoder = nn.Linear(16 * fov_size[0] * fov_size[1], 16)
        self.action_encoder = nn.Linear(action_dim, 16)
        self.value_decoder = nn.Linear(32, 1)
        self.advantage_decoder = nn.Linear(32, action_dim)

    def _get_observed_agents_from_states(self, state):
        num_agents = len(state)
        observed_agents = [[] for _ in range(num_agents)]
        for i in range(num_agents):
            observed_agents[i].append(i)
            for j in range(i + 1, num_agents):
                x_i, y_i = state[i][0], state[i][1]
                x_j, y_j = state[j][0], state[j][1]
                if abs(x_i - x_j) <= obs_r[0] and abs(y_i - y_j) <= obs_r[1]:
                    observed_agents[i].append(j)
        return observed_agents

    def forward(self, obs, action, state):
        num_agents = len(state)
        observed_agents = self._get_observed_agents_from_states(state)
        latent = []
        for i in range(num_agents):
            c_i = []
            for j in observed_agents[i]:
                o_j = [self.obs_encoder(self.fov_encoder(obs[j, k, :])) for k in range(K_obs)]
                o_j = torch.mean(torch.stack(o_j), 0)
                a_j = F.one_hot(action[j].clone().to(torch.int64), num_classes=action_dim).float()
                a_j = self.action_encoder(a_j)
                c_i.append(torch.concat((o_j, a_j)))
            c_i = torch.stack(c_i)
            latent.append(torch.sum(c_i, 0))
        V_n = [self.value_decoder(x) for x in latent]
        A_n = [self.advantage_decoder(x) for x in latent]
        Q_n = [V + A - A.mean(0, keepdim=True) for V, A in zip(V_n, A_n)]
        return Q_n
    
    def get_coma_baseline(self, obs, action, state): #value or function that approximates the expected return
        b = []
        for i in range(len(state)):
            p = 0.0 
            for j in range(self.action_dim):
                temp_action = copy.deepcopy(action)
                temp_action[i] = j
                p += self.forward(obs, action, state)[i] / self.action_dim
            b.append(p)
        return b