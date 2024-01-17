from typing import Dict, Optional
import copy
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
            state_bounds:Optional[torch.Tensor]=None,
            device=torch.device('cpu')):
        
        super().__init__()
        self.cfg = config
        self.policy = policy
        self.device = device
        self.n_dofs = self.cfg.n_dofs
        self.dt = self.cfg.control_dt
        self.state_bounds = state_bounds

        if self.state_bounds is not None:
            self.bounds_mid_range = (self.state_bounds[:,1] + self.state_bounds[:,0])/2.0
            self.bounds_half_range = (self.state_bounds[:,1] - self.state_bounds[:,0])/2.0

            self.q_pos_bounds = self.state_bounds[0:self.n_dofs]
            self.q_pos_mid_range = self.bounds_mid_range[0:self.n_dofs]
            self.q_pos_half_range = self.bounds_half_range[0:self.n_dofs]

            self.q_vel_bounds = self.state_bounds[self.n_dofs:2*self.n_dofs]
            self.q_vel_mid_range = self.bounds_mid_range[self.n_dofs:2*self.n_dofs]
            self.q_vel_half_range = self.bounds_half_range[self.n_dofs:2*self.n_dofs]

        self.state_filter = JointStateFilter(
            filter_coeff=self.cfg.state_filter_coeff, 
            # bounds=self.state_bounds,
            device=self.device,
            n_dofs=self.n_dofs,
            dt=self.dt)
        
    def forward(self, input_dict):
        return self.policy.forward(input_dict)

    def get_action(self, input_dict, deterministic=False):
        # st=time.time()
        #filter states
        input_dict['states'] = copy.deepcopy(self.state_filter.filter_joint_state(input_dict['states']))
            # dict_to_device(input_dict['states'], self.device)))
        action, _ = self.policy.get_action(input_dict, deterministic)
        command_dict = copy.deepcopy(self.state_filter.predict_internal_state(action))
        command_tensor = torch.cat([
            command_dict['q_pos'], 
            command_dict['q_vel'], 
            command_dict['q_acc']], dim=-1)
                
        return action, {'command': command_tensor, 'filtered_states': input_dict['states']} #, 'scaled_action': scaled_action}

    def log_prob(self, input_dict: Dict[str, torch.Tensor], actions:torch.Tensor):
        return self.policy.log_prob(input_dict, actions)

    def entropy(self, input_dict: Dict[str, torch.Tensor], num_samples:int = 1):
        return self.policy.entropy(input_dict, num_samples)

    def bound_joint_command(self, command_dict:Dict[str, torch.Tensor]):
        if self.state_bounds is not None:
            q_vel_before = command_dict['q_vel'].clone()
            command_dict['q_vel'] = command_dict['q_vel'].clamp(self.q_vel_bounds[:,0], self.q_vel_bounds[:,1])
            command_dict['q_pos'] = command_dict['q_pos'].clamp(self.q_pos_bounds[:,0], self.q_pos_bounds[:,1])
            if not torch.allclose(q_vel_before, command_dict['q_vel']):
                print('q_vel_clamped')

        return command_dict

    # def enforce_bounds(self, value, lows, highs):
    #     # value = torch.tanh(value)
    
    #     return value.clamp(lows, highs)

    # def enforce_bounds(self, value, mid_range, half_range):
    #     value = torch.tanh(value)

    #     return mid_range.unsqueeze(0) + value * half_range.unsqueeze(0)

    def update_task_params(self, param_dict):
        self.policy.update_task_params(param_dict)
    
    def compute_value_estimate(self, input_dict):
        return self.policy.compute_value_estimate(input_dict)

    def reset(self, reset_data):
        self.prev_qdd_des = None
        self.state_filter.reset()
        self.policy.reset(reset_data)
        # if self.task is not None:
        #     self.task.update_params(reset_data)                

