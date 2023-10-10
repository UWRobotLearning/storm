from operator import itemgetter
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
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

    def forward(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor):
        obs = obs_dict['obs']
        # act = torch.cat([act_dict[k] for k in act_dict])
        # assert obs.ndim == act_batch.ndim
        input = torch.cat([obs, actions], dim=-1)
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
    
    def all(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor):
        obs = obs_dict['obs']
        input = torch.cat([obs, actions], -1)
        return [self.net1(input), self.net2(input)]

    def forward(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor):
        # return torch.min(*self.all(obs_dict, actions))
        return torch.max(*self.all(obs_dict, actions))


class EnsembleQFunction(nn.Module):
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
        self.ensemble_size = self.config['ensemble_size']
        self.prediction_size = self.config['prediction_size']
        self.in_dim = obs_dim + act_dim
        self.out_dim = 1

        self.layer_sizes = [self.in_dim] + OmegaConf.to_object(self.mlp_params['hidden_layers']) + [self.out_dim]

        self.nets = []
        for i in range(self.ensemble_size):
            net = mlp(
            layer_sizes=self.layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'],
            squeeze_output=False)
            self.add_module('net_{}'.format(i), net)
            self.nets.append(net)

        self.to(self.device)
    
    def all(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor):
        obs = obs_dict['obs']
        input = torch.cat([obs, actions], -1)
        return torch.cat([net(input) for net in self.nets], dim=-1)

    def _idx(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor, idxs):
        obs = obs_dict['obs']
        input = torch.cat([obs, actions], -1)
        subset = [self.nets[idx] for idx in idxs]
        return torch.cat([net(input) for net in subset], dim=-1)


    def forward(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor):
        #select random unique idxs (without replacement)
        rand_idxs = np.random.choice(self.ensemble_size, size=self.prediction_size, replace=False)
        #return minimum predicted value amongst chosen members
        # return torch.min(self._idx(obs_dict, actions, rand_idxs), dim=-1)[0]
        return torch.max(self._idx(obs_dict, actions, rand_idxs), dim=-1)[0]

        # return torch.min(*self._idx(obs_dict, actions, rand_idxs))