import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, config, 
                 act_highs=None, act_lows=None, device=torch.device('cpu')):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = config
        self.device = device
        self.act_highs = act_highs
        self.act_lows = act_lows
        if self.act_highs is not None:
            self.act_half_range = (self.act_highs - self.act_lows) / 2.0
            self.act_mid_range = (self.act_highs + self.act_lows) / 2.0

    def forward(self, obs_dict):
        pass

    def get_action(self, obs_dict, deterministic=False):
        raise NotImplementedError('Policy must implement get_actin method')
    
    def scale_action(self, action:torch.Tensor): 
        #rescale action to original bounds assuming it is 
        # in the range [-1, 1]   
        if self.act_highs is not None:
            return self.act_mid_range.unsqueeze(0) + action * self.act_half_range.unsqueeze(0)
        return action
    
    def reset(self, reset_data=None):
        raise NotImplementedError('Policy must implement reset function')