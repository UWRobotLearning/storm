from typing import Optional
from storm_kit.learning.replay_buffers import ReplayBuffer
import torch
import numpy as np


class RobotBuffer(ReplayBuffer):
    def __init__(self, capacity:int, n_dofs:int, device:torch.device = torch.device('cpu')):
        self.capacity = capacity
        self.device = device

        self.q_pos_buff = torch.empty(capacity, n_dofs, device=self.device)
        self.q_vel_buff = torch.empty(capacity, n_dofs, device=self.device)
        self.q_acc_buff = torch.empty(capacity, n_dofs, device=self.device)

        self.q_pos_cmd_buff = torch.empty(capacity, n_dofs, device=self.device)
        self.q_vel_cmd_buff = torch.empty(capacity, n_dofs, device=self.device)
        self.q_acc_cmd_buff = torch.empty(capacity, n_dofs, device=self.device)


        self.curr_idx = 0
        self.num_stored = 0


    def add(self, q_pos, q_vel, q_acc, q_pos_cmd, q_vel_cmd, q_acc_cmd):

        q_pos = q_pos.to(self.device)
        q_vel = q_vel.to(self.device)
        q_acc = q_acc.to(self.device)
        q_pos_cmd = q_pos_cmd.to(self.device)
        q_vel_cmd = q_vel_cmd.to(self.device)
        q_acc_cmd = q_acc_cmd.to(self.device)


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
            # self.act_buff[0:extra] = act[-extra:]
            # self.rew_buff[0:extra] = reward[-extra:]
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

        return q_pos_batch, q_vel_batch, q_acc_batch, q_pos_cmd_batch, q_vel_cmd_batch, q_acc_cmd_batch

    def __len__(self):
        return self.num_stored
    
    def save(self, filepath):
        # state = {
        #     'q_pos_buff': self.q_pos_buff,
        #     'q_vel_buff': self.q_vel_buff,
        #     'q_acc_buff': self.q_acc_buff,
        #     'q_pos_cmd_buff': self.q_pos_cmd_buff,
        #     'q_vel_cmd_buff': self.q_vel_cmd_buff,
        #     'q_acc_cmd_buff': self.q_acc_cmd_buff,
        #     'curr_idx': self.curr_idx,
        #     'num_stored': self.num_stored
        # }
        state = self.get_as_tensor_dict()
        torch.save(state, filepath)
    
    def load(self, filepath):
        state = torch.load(filepath)
        self.q_pos_buff = state['q_pos_buff']
        self.q_vel_buff = state['q_vel_buff']
        self.q_acc_buff = state['q_acc_buff']
        self.q_pos_cmd_buff = state['q_pos_cmd_buff']
        self.q_vel_cmd_buff = state['q_vel_cmd_buff']
        self.q_acc_cmd_buff = state['q_acc_cmd_buff']

    def get_as_tensor_dict(self):
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
        return state