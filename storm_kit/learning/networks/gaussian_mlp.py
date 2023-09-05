from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from omegaconf import OmegaConf

from storm_kit.learning.networks.utils import mlp

#TODO: Add init_mean and GMM

class GaussianMLP(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            mlp_params: Dict[str, Any],
            init_std: float = 1.0,
            min_std: float = 1e-5,
            max_std: Optional[float] = 10.0,
            std_type: str = 'homoscedastic',
            learn_logvar_bounds: bool = False,
            device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        assert std_type in ['homoscedastic', 'heteroscedastic', 'fixed', 'no_std']
        self.mlp_params = mlp_params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self.std_type = std_type
        self.learn_logvar_bounds = learn_logvar_bounds
        self.device = device

        self.init_log_std = np.log(self.init_std)
        
        self.min_log_std = nn.Parameter(torch.ones(self.out_dim) * np.log(self.min_std),
                                        requires_grad = self.learn_logvar_bounds)
        if self.max_std is not None:
            self.max_log_std = nn.Parameter(torch.ones(self.out_dim) * np.log(self.max_std),
                                            requires_grad = self.learn_logvar_bounds)
        else:
            self.max_log_std = None

        if self.std_type == 'heteroscedastic':
            self.out_dim = 2*self.out_dim
            self.layer_sizes = [self.in_dim] + OmegaConf.to_object(self.mlp_params['hidden_layers']) + [self.out_dim]
            self.net = mlp(
                layer_sizes=self.layer_sizes, 
                activation=self.mlp_params['activation'],
                output_activation=self.mlp_params['output_activation'],
                dropout_prob=self.mlp_params['dropout_prob'],
                layer_norm=self.mlp_params['layer_norm'])
            #set last layer biases according to init_std
            self.net[-1].weight.data[self.out_dim//2:] *= 0.
            self.net[-1].bias.data[self.out_dim//2:] = self.init_log_std 

        elif self.std_type in ['homoscedastic', 'fixed', 'no_std']:
            self.layer_sizes = [self.in_dim] + OmegaConf.to_object(self.mlp_params['hidden_layers']) + [self.out_dim]
            self.net = mlp(
                layer_sizes=self.layer_sizes, 
                activation=self.mlp_params['activation'],
                output_activation=self.mlp_params['output_activation'],
                dropout_prob=self.mlp_params['dropout_prob'],
                layer_norm=self.mlp_params['layer_norm'])
            
            self.log_std = nn.Parameter(torch.ones(self.out_dim) * self.init_log_std)
        
        if self.std_type == 'no_std':
            self.log_std = None
        
        if self.std_type == 'fixed':
            self.log_std.requires_grad = False
            self.max_log_std.requires_grad = False
            self.min_log_std.requires_grad = False

        self.to(self.device)

    
    def forward(self, x: torch.Tensor):
        if self.std_type == 'heteroscedastic':
            mean_and_logstd = self.net(x)
            mean, logstd = mean_and_logstd.split(mean_and_logstd.shape[-1]//2, dim=-1)
        else:
            mean = self.net(x)
            logstd = self.log_std

        if logstd is not None:
            #clamp logstd
            if self.max_log_std is not None:
                logstd = self.max_log_std - F.softplus(self.max_log_std - logstd)
            logstd = self.min_log_std + F.softplus(logstd - self.min_log_std)
            std = torch.exp(logstd)
            #create normal distribution
            return mean, std
        
        return mean, None


    def sample(self, x:torch.Tensor, deterministic:bool = False, num_samples:int = 1):
        mean, std = self.forward(x)
        if deterministic or self.std_type == 'no_std':
            return mean
        
        random_samples = torch.randn(num_samples, mean.shape[0], mean.shape[1], device=self.device)
        
        return mean + std * random_samples


        # return mean + std * torch.normal(torch.zeros_like(mean), torch.ones_like(mean)) 

    def extra_repr(self):
        repr_str = '(init_std): {}\n(min_std): {}\n(max_std): {}\n(std_type): {}\n'.format(
            self.init_std, self.min_std, self.max_std, self.std_type
        )
        repr_str += '(min_log_std): {}\n(max_log_std):{}\n'.format(
            self.min_log_std, self.max_log_std
        )
        if self.std_type in ['homoscedastic', 'fixed']:
            repr_str += '(log_std): {}\n'.format(self.log_std)
        return repr_str

if __name__ == "__main__":
    mlp_params = OmegaConf.create({
        'hidden_layers': [256, 256],
        'activation': 'torch.nn.ReLU',
        'output_activation': None,
        'dropout_prob': 0.5,
        'layer_norm': False
    })


    in_dim = 4
    out_dim = 4
    model = GaussianMLP(
        in_dim=in_dim,
        out_dim=out_dim,
        mlp_params=mlp_params,
        init_std=1.0,
        std_type = 'no_std',
        learn_logvar_bounds=False,
    )
    print(model)

    inp = torch.randn(in_dim)
    mean, logvar = model(inp)
    print(mean, logvar)
    print(mean.sum())
    mean.sum().backward()
    print(model.net[-4].weight.grad)
    print(model.sample(inp))
