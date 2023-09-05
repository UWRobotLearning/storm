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
            rollout_cls=None,
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

        self.rollout = None
        if rollout_cls is not None:
            self.rollout = self.init_rollout(rollout_cls)
        self.goal_dict = None


    def forward(self, input_dict: Dict[str,torch.tensor], skip_tanh: bool =False):
        inp = self.get_policy_input(input_dict)
        mean, std = self.mlp(inp)
        dist = MultivariateNormal(mean, std)
        if self.use_tanh and not skip_tanh:
            dist = TransformedDistribution(dist, TanhTransform(cache_size=1))
        return dist

    def get_action(self, input_dict: Dict[str, torch.tensor], deterministic: bool = False, num_samples:int = 1):
        inp = self.get_policy_input(input_dict)
        act = self.mlp.sample(inp, deterministic=deterministic, num_samples=num_samples)
        if self.use_tanh:
            act = torch.tanh(act)
        action_dict = {'raw_action': act}
        return action_dict
    
    def log_prob(self, input_dict: Dict[str, torch.tensor], act_dict: Dict[str, torch.tensor]):
        dist = self.forward(input_dict)
        act = torch.cat([act_dict[k] for k in act_dict]).to(self.device)
        log_prob = dist.log_prob(act)
        if torch.any(torch.isnan(log_prob)):
            print('in log_prob calc')
            print(dist)
            import pdb; pdb.set_trace()

        return log_prob
    
    def entropy(self, input_dict: Dict[str, torch.tensor], num_samples:int = 1):
        act_dict = self.get_action(input_dict, num_samples=num_samples)
        log_prob = self.log_prob(input_dict, act_dict)
        return log_prob.sum(-1).mean(0)


    def init_rollout(self, rollout_cls):
        world_params = self.cfg.world
        # rollout_params = self.cfg.rollout
        # with open_dict(rollout_params):
        #     self.cfg['rollout']['num_instances'] = self.cfg['mppi']['num_instances']
        #     self.cfg['rollout']['horizon'] = 1
        #     self.cfg['rollout']['num_particles'] = 1

        return rollout_cls(
            cfg = self.cfg['rollout'], world_params=world_params, device=self.device)

    def update_goal(self, goal_dict):
        self.goal_dict = goal_dict
        if self.rollout is not None:
            self.rollout.update_params(goal_dict=goal_dict)
    
    def get_policy_input(self, input_dict):
        state = input_dict['state'] if 'state' in input_dict else None
        if self.rollout is not None and state is not None:
            obs, _ = self.rollout.compute_observations(state_dict=state)
        else:
            obs = input_dict['obs']
        return obs

    def extra_repr(self):
        repr_str = '(use_tanh): {}\n'.format(self.use_tanh)
        return repr_str

    def reset(self, reset_data):
        pass

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
