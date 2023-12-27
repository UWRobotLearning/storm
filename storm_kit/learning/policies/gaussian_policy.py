from typing import Dict, Any
import torch
from torch.distributions import Normal, MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from storm_kit.learning.policies import Policy
from storm_kit.learning.networks import GaussianMLP


class GaussianPolicy(Policy):
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            config,
            task, 
            act_highs=None,
            act_lows=None,
            device: torch.device = torch.device('cpu'),
            ):
        
        super().__init__(obs_dim, act_dim, config, device)
        self.mlp_params = self.cfg['mlp_params']
        self.init_std = self.cfg['init_std']
        self.min_std = self.cfg['min_std']
        self.max_std = self.cfg['max_std']
        self.std_type = self.cfg['std_type']
        self.learn_logvar_bounds = self.cfg['learn_logvar_bounds']
        self.use_tanh = self.cfg['use_tanh']
        self.act_highs = act_highs
        self.act_lows = act_lows
        self.rescale_action = self.use_tanh
        self.task = task

        self.mlp = GaussianMLP(
            in_dim = self.obs_dim,
            out_dim = self.act_dim,
            mlp_params = self.mlp_params,
            init_std = self.init_std,
            min_std = self.min_std,
            max_std = self.max_std,
            std_type = self.std_type,
            learn_logvar_bounds = self.learn_logvar_bounds,
            device = self.device
        )

    def forward(self, input_dict: Dict[str,torch.Tensor], skip_tanh: bool =False):
        
        if 'obs' in input_dict:
            inp = input_dict['obs']
        else:
            if self.task is not None:
                inp = self.task.compute_observations(
                    input_dict['states'], compute_full_state=True)

        mean, std = self.mlp(inp)
        dist = Normal(mean, std)
        if self.use_tanh and not skip_tanh:
            dist = TransformedDistribution(dist, TanhTransform(cache_size=1))
        return dist

    def get_action(self, input_dict: Dict[str, torch.Tensor], deterministic: bool = False): #, num_samples:int = 1):
        if 'obs' in input_dict:
            inp = input_dict['obs']
        else:
            if self.task is not None:
                inp = self.task.compute_observations(
                    input_dict['states'], compute_full_state=True)
        act = self.mlp.sample(inp, deterministic=deterministic) #, num_samples=num_samples)
        if self.use_tanh:
            act = torch.tanh(act)
        if self.rescale_action:
            act = self.scale_action(act)
        return act
    
    def log_prob(self, input_dict: Dict[str, torch.Tensor], actions: torch.Tensor):
        dist = self.forward(input_dict)
        log_prob = dist.log_prob(actions)
        return log_prob
    
    def entropy(self, input_dict: Dict[str, torch.Tensor], num_samples:int = 1):
        dist = self.forward(input_dict)
        actions = dist.rsample(sample_shape=torch.Size([num_samples]))
        log_prob = dist.log_prob(actions)
        return actions, log_prob.mean(0)

    def reset(self, reset_data):
        if self.task is not None:
            self.task.update_params(reset_data)                

    def extra_repr(self):
        repr_str = '(use_tanh): {}\n'.format(self.use_tanh)
        return repr_str

    def scale_action(self, action:torch.Tensor):
        act_half_range = (self.act_highs - self.act_lows) / 2.0
        act_mid_range = (self.act_highs + self.act_lows) / 2.0
    
        return act_mid_range.unsqueeze(0) + action * act_half_range.unsqueeze(0)


if __name__ == "__main__":
    import torch.nn as nn

    mlp_params = {
        'hidden_layers': [256, 256],
        'activation': nn.ReLU,
        'output_activation': None,
        'dropout_prob': 0.5,
        'layer_norm': False,
    }
    obs_dim = 4
    act_dim = 4
    config = {
        'mlp_params': mlp_params,
        'init_std': 1.0,
        'min_std': 1e-5,
        'max_std': 10,
        'std_type': 'heteroscedastic',
        'learn_logvar_bounds': True,
        'use_tanh': True
    }

    policy = GaussianPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        config=config
    )
    print(policy)

    obs = torch.randn(obs_dim)
    dist = policy(obs)
    print(dist)
    print('random action')
    action = policy.get_action(obs, determinisitc=False)
    print(action)
    print('det. action')
    action = policy.get_action(obs, determinisitc=True)
    print(action)
