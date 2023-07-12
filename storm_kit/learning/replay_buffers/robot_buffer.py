from typing import Optional
from storm_kit.learning.replay_buffers import ReplayBuffer
import torch
import numpy as np


class RobotBuffer(ReplayBuffer):
    def __init__(self, capacity:int, n_dofs:int, obs_dim:Optional[int]=None, act_dim:Optional[int]=None, device:torch.device = torch.device('cpu')):
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim
        

        self.q_pos_buff = torch.empty(capacity, n_dofs, device=self.device)
        self.q_vel_buff = torch.empty(capacity, n_dofs, device=self.device)
        self.q_acc_buff = torch.empty(capacity, n_dofs, device=self.device)

        self.q_pos_cmd_buff = torch.empty(capacity, n_dofs, device=self.device)
        self.q_vel_cmd_buff = torch.empty(capacity, n_dofs, device=self.device)
        self.q_acc_cmd_buff = torch.empty(capacity, n_dofs, device=self.device)

        self.ee_goal_buff = torch.empty(capacity, 7, device=self.device)
        self.joint_goal_buff = torch.empty(capacity, n_dofs, device=self.device)
        self.cost_buff = torch.empty(capacity, device=self.device)
        
        if self.obs_dim is not None:
            self.obs_buff = torch.empty(capacity, self.obs_dim, device=self.device)
        
        self.ee_goal_stored = False
        self.joint_goal_stored = False
        self.costs_stored = False
        self.obs_stored = False
        
        self.curr_idx = 0
        self.num_stored = 0


    def add(self, 
            q_pos, q_vel, q_acc, 
            q_pos_cmd, q_vel_cmd, q_acc_cmd, 
            ee_goal=None, joint_goal=None, 
            obs=None, reward=None):

        q_pos = q_pos.to(self.device)
        q_vel = q_vel.to(self.device)
        q_acc = q_acc.to(self.device)
        q_pos_cmd = q_pos_cmd.to(self.device)
        q_vel_cmd = q_vel_cmd.to(self.device)
        q_acc_cmd = q_acc_cmd.to(self.device)
        if ee_goal is not None:
            ee_goal = ee_goal.to(self.device)
            self.ee_goal_stored = True
        if joint_goal is not None:
            joint_goal = joint_goal.to(self.device)
            self.joint_goal_stored = True
        if obs is not None:
            self.obs_stored = True
            obs = obs.to(self.device)
        if reward is not None:
            self.costs_stored = True

        num_obs = q_pos.shape[0]
        remaining = min(self.capacity - self.curr_idx, num_obs)
        if num_obs > remaining:
            #add to front
            extra = num_obs - remaining
            self.q_pos_buff[0:extra] = q_pos[-extra:]
            self.q_vel_buff[0:extra] = q_vel[-extra:]
            self.q_acc_buff[0:extra] = q_acc[-extra:]
            self.q_pos_cmd_buff[0:extra] = q_pos_cmd[-extra:]
            self.q_vel_cmd_buff[0:extra] = q_vel_cmd[-extra:]
            self.q_acc_cmd_buff[0:extra] = q_acc_cmd[-extra:]
            if ee_goal is not None:
                self.ee_goal_buff[0:extra] = ee_goal[-extra:]
            if joint_goal is not None:
                self.joint_goal_buff[0:extra] = joint_goal[-extra:]
            if obs is not None:
                self.obs_buff[0:extra] = obs[-extra:]
            if reward is not None:
                self.cost_buff[0:extra] = reward[-extra:]

            # self.act_buff[0:extra] = act[-extra:]
            # self.cost_buff[0:extra] = reward[-extra:]
            # self.next_obs_buff[0:extra] = next_obs[-extra:]
            # self.done_buff[0:extra] = done[-extra:]
            # if state is not None:
            #     self.state_buff[0:extra] = state[-extra:]
            #     self.next_state_buff[0:extra] = next_state[-extra:]

        #add to end
        self.q_pos_buff[self.curr_idx:self.curr_idx + remaining] = q_pos[0:remaining]
        self.q_vel_buff[self.curr_idx:self.curr_idx + remaining] = q_vel[0:remaining]
        self.q_acc_buff[self.curr_idx:self.curr_idx + remaining] = q_acc[0:remaining]
        self.q_pos_cmd_buff[self.curr_idx:self.curr_idx + remaining] = q_pos_cmd[0:remaining]
        self.q_vel_cmd_buff[self.curr_idx:self.curr_idx + remaining] = q_vel_cmd[0:remaining]
        self.q_acc_cmd_buff[self.curr_idx:self.curr_idx + remaining] = q_acc_cmd[0:remaining]
        if ee_goal is not None:
            self.ee_goal_buff[self.curr_idx:self.curr_idx + remaining] = ee_goal[0:remaining]
        if joint_goal is not None:
            self.joint_goal_buff[self.curr_idx:self.curr_idx + remaining] = joint_goal[0:remaining]
        if obs is not None:
            self.obs_buff[self.curr_idx:self.curr_idx + remaining] = obs[0:remaining]
        if reward is not None:
            self.cost_buff[self.curr_idx:self.curr_idx + remaining] = reward[0:remaining]

        self.curr_idx = (self.curr_idx + num_obs) % self.capacity
        self.num_stored = min(self.num_stored + num_obs, self.capacity)
    
    def sample(self, batch_size):
        idxs = torch.randint(0, len(self), size=(batch_size,), device=self.device)
        q_pos_batch = self.q_pos_buff[idxs]
        q_vel_batch = self.q_vel_buff[idxs]
        q_acc_batch = self.q_acc_buff[idxs]
        q_pos_cmd_batch = self.q_pos_cmd_buff[idxs]
        q_vel_cmd_batch = self.q_vel_cmd_buff[idxs]
        q_acc_cmd_batch = self.q_acc_cmd_buff[idxs]
        ee_goal_batch, joint_goal_batch = None, None
        obs_batch, rew_batch = None, None
        if self.ee_goal_stored:
            ee_goal_batch = self.ee_goal_buff[idxs]
        if self.joint_goal_stored:
            joint_goal_batch = self.joint_goal_buff[idxs]
        if self.obs_stored:
            obs_batch = self.obs_buff[idxs]
        if self.costs_stored:
            rew_batch = self.cost_buff[idxs]

        batch = {
            'q_pos': q_pos_batch,
            'q_vel': q_vel_batch,
            'q_acc': q_acc_batch,
            'q_pos_cmd': q_pos_cmd_batch,
            'q_vel_cmd': q_vel_cmd_batch,
            'q_acc_cmd': q_acc_cmd_batch,
            'ee_goal': ee_goal_batch,
            'joint_goal': joint_goal_batch,
            'obs': obs_batch,
            'rew': rew_batch
        }

        return batch

    def __len__(self):
        return self.num_stored

    def concatenate(self, state_dict):

        q_pos = state_dict['q_pos_buff']
        q_vel = state_dict['q_vel_buff']
        q_acc = state_dict['q_acc_buff']
        q_pos_cmd = state_dict['q_pos_cmd_buff']
        q_vel_cmd = state_dict['q_vel_cmd_buff']
        q_acc_cmd = state_dict['q_acc_cmd_buff']
        ee_goal, joint_goal = None, None
        ee_goal = state_dict['ee_goal_buff'] if 'ee_goal_buff' in state_dict else None
        joint_goal = state_dict['joint_goal_buff'] if 'joint_goal_buff' in state_dict else None
        obs = state_dict['obs_buff'] if 'obs_bufd' in state_dict else None
        reward = state_dict['cost_buff'] if 'cost_buff' in state_dict else None

        self.add(q_pos, q_vel, q_acc, q_pos_cmd, q_vel_cmd, q_acc_cmd, ee_goal, joint_goal, obs, reward)
    
    def save(self, filepath):
        state = self.state_dict()
        torch.save(state, filepath)
    
    def load(self, filepath):
        state = torch.load(filepath)
        self.q_pos_buff = state['q_pos_buff'].to(self.device)
        self.q_vel_buff = state['q_vel_buff'].to(self.device)
        self.q_acc_buff = state['q_acc_buff'].to(self.device)
        self.q_pos_cmd_buff = state['q_pos_cmd_buff'].to(self.device)
        self.q_vel_cmd_buff = state['q_vel_cmd_buff'].to(self.device)
        self.q_acc_cmd_buff = state['q_acc_cmd_buff'].to(self.device)
        if 'ee_goal_buff' in state:
            self.ee_goal_buff = state['ee_goal_buff'].to(self.device)
            self.ee_goal_stored = True
        if 'joint_goal_buff' in state:
            self.joint_goal_buff = state['joint_goal_buff'].to(self.device)
            self.joint_goal_stored = True
        if 'obs_buff' in state:
            self.obs_buff = state['obs_buff'].to(self.device)
            self.obs_stored = True
        if 'cost_buff' in state:
            self.cost_buff = state['cost_buff'].to(self.device)
            self.costs_stored = True

        self.capacity = state['num_stored']
        self.num_stored = state['num_stored']


    def state_dict(self):
        state = {
            'q_pos_buff': self.q_pos_buff[0:self.num_stored],
            'q_vel_buff': self.q_vel_buff[0:self.num_stored],
            'q_acc_buff': self.q_acc_buff[0:self.num_stored],
            'q_pos_cmd_buff': self.q_pos_cmd_buff[0:self.num_stored],
            'q_vel_cmd_buff': self.q_vel_cmd_buff[0:self.num_stored],
            'q_acc_cmd_buff': self.q_acc_cmd_buff[0:self.num_stored],
            'curr_idx': self.curr_idx,
            'num_stored': self.num_stored
        }
        if self.ee_goal_stored:         
            state['ee_goal_buff'] = self.ee_goal_buff[0:self.num_stored]
        if self.joint_goal_stored:         
            state['joint_goal_buff'] = self.joint_goal_buff[0:self.num_stored]
        if self.obs_stored:
            state['obs_buff'] = self.obs_buff[0:self.num_stored]
        if self.costs_stored:
            state['cost_buff'] = self.cost_buff[0:self.num_stored]
        return state

    def __repr__(self):
        str = 'num_stored={}, capacity={}, ee_goal_stored={}, joint_goal_stored={}, obs_stored={}, rew_stored={}'.format(
            self.num_stored, self.capacity, self.ee_goal_stored, self.joint_goal_stored, self.obs_stored, self.costs_stored)
        return str