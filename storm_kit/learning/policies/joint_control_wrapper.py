from typing import Dict
import copy
from omegaconf import open_dict
import torch
import torch.nn as nn
import time

from torch.profiler import profile, record_function, ProfilerActivity


from storm_kit.learning.policies import Policy, GaussianPolicy
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.rollout.arm_reacher import ArmReacher
from storm_kit.mpc.utils.state_filter import JointStateFilter

class JointControlWrapper(nn.Module):
    def __init__(
            self, 
            config, 
            policy,
            device=torch.device('cpu')):
        super().__init__()
        self.cfg = config
        self.policy = policy
        self.device = device
        self.n_dofs = self.cfg.rollout.n_dofs

        # self.rollout = self.init_rollout(rollout_cls) 
        self.dt = self.cfg.rollout.control_dt

        self.state_filter = JointStateFilter(
            filter_coeff=self.cfg.rollout.state_filter_coeff, 
            device=self.device,
            n_dofs=self.n_dofs,
            dt=self.dt)
        
        self.prev_qdd_des = None
        self.goal_dict = None 


    def forward(self, input_dict):
        # state_dict = input_dict['states']
        # states = torch.cat((
        #     state_dict['q_pos'], state_dict['q_vel'], state_dict['q_acc'], state_dict['tstep']), dim=-1)
        # states = states.to(self.device)
        # self.state_filter.predict_internal_state(self.prev_qdd_des)
        
        # planning_state = self.state_filter.filter_joint_state(states)

        # policy_input = copy.deepcopy(input_dict)
        # policy_input['states']['q_pos'] = planning_state[:, 0:self.n_dofs]
        # policy_input['states']['q_vel'] = planning_state[:, self.n_dofs:2*self.n_dofs]
        # policy_input['states']['q_acc'] = planning_state[:, 2*self.n_dofs:3*self.n_dofs]
        
        # curr_action_seq = self.policy.forward(input_dict)
        
        # self.prev_qdd_des = q_acc_des.clone()
        raw_distribution = self.policy.forward(input_dict, skip_tanh=True)



        return self.policy.forward(input_dict)


    def get_action(self, input_dict, deterministic=False, num_samples:int = 1):
        state_dict = input_dict['states']
        states = torch.cat((state_dict['q_pos'], state_dict['q_vel'], state_dict['q_acc'], state_dict['tstep']), dim=-1)
        states = states.to(self.device)
        # self.state_filter.predict_internal_state(self.prev_qdd_des)

        # # if self.state_filter.cmd_joint_state is None:
        # #     state_dict['velocity'] *= 0.0
        # planning_state = self.state_filter.filter_joint_state(states)

        # policy_input = input_dict
        # policy_input['states'] = planning_state

        # curr_action = self.policy.forward(policy_input)
        action = self.policy.get_action(input_dict)['raw_action']

        q_acc_des = action
        q_vel_des = states[:, self.n_dofs:2*self.n_dofs] + q_acc_des * self.dt
        q_pos_des = states[:, :self.n_dofs] + q_vel_des * self.dt

        command = {
            'q_pos_des': q_pos_des,
            'q_vel_des': q_vel_des,
            'q_acc_des': q_acc_des,
        }

        self.prev_qdd_des = q_acc_des.clone()
        return command

    def log_prob(self, input_dict: Dict[str, torch.Tensor], act_dict: Dict[str, torch.Tensor]):
        dist = self.forward(input_dict)
        act = torch.cat([act_dict[k] for k in act_dict], dim=-1).to(self.device)
        log_prob = dist.log_prob(act)
        return log_prob
    
    def entropy(self, input_dict: Dict[str, torch.Tensor], num_samples:int = 1):
        act_dict = self.get_action(input_dict, num_samples=num_samples)
        log_prob = self.log_prob(input_dict, act_dict)
        return log_prob.sum(-1).mean(0)


    def update_goal(self, goal_dict):
        self.goal_dict = goal_dict
        self.policy.update_goal(goal_dict)

    def reset(self):
        self.prev_qdd_des = None
        self.state_filter.reset()
