from operator import itemgetter
from omegaconf import OmegaConf
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from storm_kit.learning.networks.utils import mlp, ensemble_mlp
from storm_kit.learning.learning_utils import VectorizedLinear

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
            squeeze_output=False)
        self.to(self.device)

    def all(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor):
        obs = obs_dict['obs']
        input = torch.cat([obs, actions], -1)
        preds =  self.net(input)
        return preds


    def forward(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor):
        preds = self.all(obs_dict, actions).squeeze(-1)
        return preds


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
        self.aggregation = self.config.get('aggregation', 'max')

        self.layer_sizes = [self.in_dim] + OmegaConf.to_object(self.mlp_params['hidden_layers']) + [self.out_dim]
        self.net1 = mlp(
            layer_sizes=self.layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'],
            squeeze_output=False).to(self.device)

        self.net2 = mlp(
            layer_sizes=self.layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'],
            squeeze_output=False)
        self.to(self.device)
    
    def all(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor):
        obs = obs_dict['obs']
        input = torch.cat([obs, actions], -1)
        preds =  torch.cat([self.net1(input), self.net2(input)], dim=-1)
        return preds


    def forward(self, obs_dict: Dict[str,torch.Tensor], actions: torch.Tensor):
        if self.aggregation == 'max':
            preds = torch.max(self.all(obs_dict, actions), dim=-1)[0]
        elif self.aggregation == 'min':
            preds = torch.min(self.all(obs_dict, actions), dim=-1)[0]
        elif self.aggregation == 'mean':
            preds = torch.mean(self.all(obs_dict, actions), dim=-1)
        return preds


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
        self.aggregation = self.config.get('aggregation', 'max')
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
        if self.aggregation == 'max':
            preds = torch.max(self._idx(obs_dict, actions, rand_idxs), dim=-1)[0]
        elif self.aggregation == 'min':
            preds = torch.min(self._idx(obs_dict, actions, rand_idxs), dim=-1)[0]
        elif self.aggregation == 'mean':
            preds = torch.mean(self._idx(obs_dict, actions, rand_idxs), dim=-1)
        return preds

class ValueFunction(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            config,
            device: torch.device = torch.device('cpu'),

    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.config = config 
        self.device = device
        self.mlp_params = self.config['mlp_params']
        self.out_dim = 1
        self.mlp_params = self.config['mlp_params']
        self.ensemble_size = self.config['ensemble_size']
        self.aggregation = self.config.get('aggregation', 'max')

        self.layer_sizes = [self.obs_dim] + OmegaConf.to_object(self.mlp_params['hidden_layers']) + [self.out_dim]
        self.net = mlp(
            layer_sizes=self.layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'],
            squeeze_output=True).to(self.device)
        
    def forward(self, obs_dict:Dict[str, torch.Tensor]):
        input = obs_dict['obs']
        return self.net(input)


class EnsembleValueFunction(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            config,
            device: torch.device = torch.device('cpu'),

    ):
        super().__init__()
        self.obs_dim = obs_dim

        self.config = config 
        self.device = device
        self.mlp_params = self.config['mlp_params']
        self.ensemble_size = self.config['ensemble_size']
        self.out_dim = 1
        self.hidden_dim = self.mlp_params['hidden_layers'][0]
        self.aggregation = self.config['aggregation']
        self.prior_factor = self.config['prior_factor']
        self.activation = self.mlp_params['activation']
        self.w_init = self.config['w_init']
        self.layer_sizes = [self.obs_dim] + OmegaConf.to_object(self.mlp_params['hidden_layers']) + [self.out_dim]

        self.net = ensemble_mlp(
            ensemble_size=self.ensemble_size,
            layer_sizes=self.layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'],
            squeeze_output=False).to(self.device)
        self.net.apply(lambda m: self.initialize_parameters(m))
    
        if self.prior_factor > 0.:
            self.prior_net = ensemble_mlp(
            ensemble_size=self.ensemble_size,
            layer_sizes=self.layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'],
            squeeze_output=False).to(self.device).requires_grad_(False)
            self.prior_net.apply(lambda m: self.initialize_parameters(m))

        if self.w_init > 0:
            last_linear = -2 if self.mlp_params['output_activation'] is not None else -1
            self.net[last_linear].weight.data.uniform_(-self.w_init, self.w_init)
            self.net[last_linear].bias.data.fill_(0.0)
            # torch.nn.init.uniform_(self.net[-2].bias, -3e-3, 3e-3)

        self.set_normalization_stats()
        # # # init as in the EDAC paper
        # for layer in self.net[::2]:
        #     torch.nn.init.constant_(layer.bias, 0.1)

        # torch.nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        # torch.nn.init.uniform_(self.net[-2].bias, -3e-3, 3e-3)

    @torch.no_grad()
    def initialize_parameters(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, VectorizedLinear):
            if self.activation == "torch.nn.Sigmoid":
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.activation == "torch.nn.Tanh":
                torch.nn.init.xavier_normal_(m.weight)
            if self.activation == "torch.nn.SiLU":
                torch.nn.init.xavier_uniform_(m.weight)
    

    # def all(self, obs_dict: Dict[str,torch.Tensor]):
    def all(self, obs:torch.Tensor, denormalized:bool=False):
        if obs.dim() != 3:
            assert obs.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            obs = obs.unsqueeze(0).repeat_interleave(
                self.ensemble_size, dim=0
            )
        assert obs.dim() == 3
        assert obs.shape[0] == self.ensemble_size
        
        obs = self.normalize_inputs(obs)
        
        # [num_critics, batch_size]
        values = self.net(obs).squeeze(-1) #inference

        if self.prior_factor > 0.:
            values += self.prior_factor * self.prior_net(obs).squeeze(-1) #inference

        #denormalize
        if denormalized:
            values = self.denormalize_predictions(values)

        info = {'mean': values.mean(dim=0), 'std': values.std(dim=0)}

        return values, info


    def forward(self, obs: torch.Tensor, denormalized:bool=False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # [..., batch_size, state_dim + action_dim]
        values, info = self.all(obs, denormalized=denormalized)
        if self.aggregation == 'max':
            preds = torch.max(values, dim=0)[0]
        elif self.aggregation == 'min':
            preds = torch.min(values, dim=0)[0]
        elif self.aggregation == 'mean':
            preds = torch.mean(values, dim=0)
        elif self.aggregation == 'median':
            preds = torch.median(values, dim=0)[0]
        
        return preds, info
 

    @torch.no_grad()
    def set_normalization_stats(self, normalization_stats=None):
        self.V_min, self.V_max=-float('inf'), float('inf')
        self.obs_min, self.obs_max=-float('inf'), float('inf')
        self.V_mean, self.V_std = 0.0, 1.0
        self.obs_mean, self.obs_std = 0.0, 1.0
        if normalization_stats is not None:
            self.V_max = normalization_stats['V_max'] if 'V_max' in normalization_stats else float('inf')
            self.V_min = normalization_stats['V_min'] if 'V_min' in normalization_stats else float('-inf')
            self.V_mean = normalization_stats['disc_return_mean'] if 'disc_return_mean' in normalization_stats else 0.0
            self.V_std = normalization_stats['disc_return_std'] if 'disc_return_std' in normalization_stats else 1.0
            self.obs_mean = normalization_stats['obs_mean'] if 'obs_mean' in normalization_stats else None
            self.obs_std = normalization_stats['obs_std'] if 'obs_std' in normalization_stats else None
            self.obs_max = normalization_stats['obs_max'] if 'obs_max' in normalization_stats else float('inf')
            self.obs_min = normalization_stats['obs_min'] if 'obs_min' in normalization_stats else float('-inf')

    @torch.no_grad()
    def normalize_predictions(self, v_preds):
        #normalize [V_min, V_max] -> [-1,1] 
        V_range = (self.V_max - self.V_min) + 1e-12
        v_preds = 2.0 * ((v_preds - self.V_min) / V_range) - 1
        return v_preds 

    @torch.no_grad()
    def denormalize_predictions(self, v_preds):
        #unnormalize [-1,1] -> [V_min, V_max]
        V_range = (self.V_max - self.V_min)
        v_preds = 0.5 * (v_preds + 1) * V_range  + self.V_min

        return v_preds

    @torch.no_grad()
    def normalize_inputs(self, obs:torch.Tensor):
        obs_range = (self.obs_max - self.obs_min) + 1e-12
        # obs = (obs - self.obs_min) / obs_range 
        obs = 2.0 * ((obs - self.obs_min) / obs_range) - 1 

        return obs

