from typing import Dict
import torch
import torch.nn as nn
import time

from storm_kit.mpc.utils.state_filter import JointStateFilter
from storm_kit.learning.learning_utils import dict_to_device

class JointControlWrapper(nn.Module):
    def __init__(
            self, 
            config, 
            policy,
            # act_highs=None,
            # act_lows=None,
            device=torch.device('cpu')):
        
        super().__init__()
        self.cfg = config
        self.policy = policy
        self.device = device
        self.n_dofs = self.cfg.n_dofs
        self.dt = self.cfg.control_dt
        # self.act_highs = act_highs
        # self.act_lows = act_lows

        self.state_filter = JointStateFilter(
            filter_coeff=self.cfg.state_filter_coeff, 
            device=self.device,
            n_dofs=self.n_dofs,
            dt=self.dt)
        
        self.prev_qdd_des = None
        self.goal_dict = None 

    def forward(self, input_dict):
        return self.policy.forward(input_dict)

    def get_action(self, input_dict, deterministic=False, num_samples:int = 1):
        state_dict = dict_to_device(input_dict['states'], self.device)
        planning_states = self.state_filter.filter_joint_state(state_dict)
        new_input_dict = {'states': planning_states}
        if 'obs' in input_dict:
            new_input_dict['obs'] = input_dict['obs']

        action = self.policy.get_action(new_input_dict, deterministic, num_samples)

        #TODO: Thjis must go by making state_filter compatible with num actions
        if action.ndim > 2:
            action = action[0]

        
        # scaled_action = action
        # if self.act_highs is not None:
        #     scaled_action = self.scale_action(action)
        
        command_dict = self.state_filter.predict_internal_state(action)

        command_tensor = torch.cat(
            [command_dict['q_pos'], command_dict['q_vel'], 
            command_dict['q_acc']], dim=-1)
        
        self.prev_qdd_des = action.clone()
        
        return command_tensor, {'action': action} #, 'scaled_action': scaled_action}

    def log_prob(self, input_dict: Dict[str, torch.Tensor], actions:torch.Tensor):
        return self.policy.log_prob(input_dict, actions)

    def entropy(self, input_dict: Dict[str, torch.Tensor], num_samples:int = 1):
        return self.policy.entropy(input_dict, num_samples)

    def update_rollout_params(self, param_dict):
        self.policy.update_rollout_params(param_dict)
    
    def compute_value_estimate(self, input_dict):
        return self.policy.compute_value_estimate(input_dict)

    def reset(self, reset_data):
        self.prev_qdd_des = None
        self.state_filter.reset()
        self.policy.reset(reset_data)
    
    def scale_action(self, action:torch.Tensor):
        act_half_range = (self.act_highs - self.act_lows) / 2.0
        act_mid_range = (self.act_highs + self.act_lows) / 2.0
    
        return act_mid_range.unsqueeze(0) + action * act_half_range.unsqueeze(0)

