from omegaconf import OmegaConf
import torch
import torch.nn as nn
from typing import Dict
from storm_kit.learning.networks.utils import mlp

class QFunction(nn.Module):
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
        self.mlp_params = self.config['mlp_params']
        self.in_dim = obs_dim + act_dim
        self.out_dim = 1

        self.layer_sizes = [self.in_dim] + OmegaConf.to_object(self.mlp_params['hidden_layers']) + [self.out_dim]
        self.net = mlp(
            layer_sizes=self.layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'],
            squeeze_output=True)
        self.to(self.device)

    def forward(self, obs_dict: Dict[str,torch.tensor], act_dict: Dict[str, torch.tensor]):
        obs = obs_dict['obs']
        act = torch.cat([act_dict[k] for k in act_dict])
        # assert obs.ndim == act_batch.ndim
        input = torch.cat([obs, act], dim=-1)
        q_pred = self.net(input)
        return q_pred



class TwinQFunction(nn.Module):
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
        self.mlp_params = self.config['mlp_params']
        self.in_dim = obs_dim + act_dim
        self.out_dim = 1

        self.layer_sizes = [self.in_dim] + OmegaConf.to_object(self.mlp_params['hidden_layers']) + [self.out_dim]
        self.net1 = mlp(
            layer_sizes=self.layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'],
            squeeze_output=True)

        self.net2 = mlp(
            layer_sizes=self.layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'],
            squeeze_output=True)
        self.to(self.device)
    
    def both(self, obs_dict: Dict[str,torch.tensor], act_dict: Dict[str, torch.tensor]):
        obs = obs_dict['obs']
        act = torch.cat([act_dict[k] for k in act_dict], dim=-1)
        input = torch.cat([obs, act], -1)
        return self.net1(input), self.net2(input)

    def forward(self, obs_dict: Dict[str,torch.tensor], act_dict: Dict[str, torch.tensor]):
        return torch.min(*self.both(obs_dict, act_dict))