import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.device = device
        self.learn_reward = self.config['learn_reward']
        self.learn_termination = self.config['learn_termination']

        self.input_dim = self.obs_dim + self.act_dim
        self.output_dim = self.obs_dim + int(self.learn_reward) + int(self.learn_termination)


    def forward(self, obs, act):
        pass

    def predict(self, obs, act, deterministic=False):
        raise NotImplementedError
    