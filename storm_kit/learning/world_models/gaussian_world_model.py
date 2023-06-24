from typing import Dict, Any
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from storm_kit.rl.world_models import WorldModel
from storm_kit.rl.networks import GaussianMLP


class GaussianWorldModel(WorldModel):
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            config,
            device: torch.device = torch.device('cpu'),
            ):
        
        super().__init__(obs_dim, act_dim, config, device)
        self.mlp_params = self.config['mlp_params']
        self.init_std = self.config['init_std']
        self.min_std = self.config['min_std']
        self.max_std = self.config['max_std']
        self.std_type = self.config['std_type']
        self.learn_logvar_bounds = self.config['learn_logvar_bounds']

        self.mlp = GaussianMLP(
            in_dim = self.input_dim,
            out_dim = self.output_dim,
            mlp_params = self.mlp_params,
            init_std = self.init_std,
            min_std = self.min_std,
            max_std = self.max_std,
            std_type = self.std_type,
            learn_logvar_bounds = self.learn_logvar_bounds,
            device = self.device
        )


    def forward(self, obs_dict: Dict[str,torch.Tensor], act: torch.Tensor):
        obs = obs_dict['obs']
        inp = torch.cat((obs, act), dim=-1)
        mean, std = self.mlp(inp)
        dist = Normal(mean, std)
        return dist

    def predict(self, obs_dict: Dict[str, torch.Tensor], act: torch.Tensor, deterministic: bool = False, num_samples:int = 1):
        # obs = obs_dict['obs']
        # inp = torch.cat((obs, act), dim=-1)
        dist = self.forward(obs_dict, act)
        preds = dist.rsample(torch.Size([num_samples]))
        # pred = self.mlp.sample(inp, deterministic=deterministic, num_samples=num_samples)
        return preds


    def extra_repr(self):
        repr_str = '(learn_reward): {}\n(learn_terminal)'.format(
            self.learn_reward, self.learn_termination)
        return repr_str

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
