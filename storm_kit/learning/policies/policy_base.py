import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, config, device=torch.device('cpu')):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = config
        self.device = device


    def forward(self, obs):
        pass

    def get_action(self, obs, deterministic=False):
        raise NotImplementedError
    
    def update_goal(self, goal):
        pass