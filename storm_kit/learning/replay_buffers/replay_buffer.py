from typing import Optional
import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity:int, obs_dim, act_dim, state_shape=None, device:torch.device = torch.device('cpu')):
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buff = torch.empty(capacity, obs_dim, device=self.device)
        self.next_obs_buff = torch.empty(capacity, obs_dim, device=self.device)
        self.act_buff = torch.empty(capacity, act_dim, device=self.device)
        
        self.state_buff = None
        self.next_state_buff = None
        if state_shape is not None:
            self.state_buff = torch.empty((capacity, state_shape), device=self.device)
            self.next_state_buff = torch.empty((capacity, state_shape), device=self.device)
        self.rew_buff = torch.empty((capacity,), device=self.device)
        self.done_buff = torch.empty((capacity,), device=self.device, dtype=torch.bool)
        self.curr_idx = 0
        self.num_stored = 0


    def add(self, obs, act, reward, next_obs, done, state=None, next_state=None):
        obs = obs.to(self.device)
        act = act.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)
        if state is not None:
            state = state.to(self.device)
            next_state = next_state.to(self.device)
         
        num_obs = obs.shape[0]
        remaining = min(self.capacity - self.curr_idx, num_obs)
        if num_obs > remaining:
            #add to front
            extra = num_obs - remaining
            self.obs_buff[0:extra] = obs[-extra:]
            self.act_buff[0:extra] = act[-extra:]
            self.rew_buff[0:extra] = reward[-extra:]
            self.next_obs_buff[0:extra] = next_obs[-extra:]
            self.done_buff[0:extra] = done[-extra:]
            if state is not None:
                self.state_buff[0:extra] = state[-extra:]
                self.next_state_buff[0:extra] = next_state[-extra:]

        #add to end
        self.obs_buff[self.curr_idx:self.curr_idx + remaining] = obs[0:remaining]
        self.act_buff[self.curr_idx:self.curr_idx + remaining] = act[0:remaining]
        self.rew_buff[self.curr_idx:self.curr_idx + remaining] = reward[0:remaining]
        self.next_obs_buff[self.curr_idx:self.curr_idx + remaining] = next_obs[0:remaining]
        self.done_buff[self.curr_idx:self.curr_idx + remaining] = done[0:remaining]
        if state is not None:
            self.state_buff[self.curr_idx:self.curr_idx + remaining] = state[0:remaining]
            self.next_state_buff[self.curr_idx:self.curr_idx + remaining] = next_state[0:remaining]        

        self.curr_idx = (self.curr_idx + num_obs) % self.capacity
        self.num_stored = min(self.num_stored + num_obs, self.capacity)
    
        
    
    def sample(self, batch_size):
        idxs = torch.randint(0, len(self), size=(batch_size,), device=self.device)
        obs_batch = self.obs_buff[idxs]
        act_batch = self.act_buff[idxs]
        rew_batch = self.rew_buff[idxs]
        next_obs_batch = self.next_obs_buff[idxs]
        done_batch = self.done_buff[idxs]
        state_batch = None
        next_state_batch = None
        if self.state_buff is not None:
            state_batch = self.state_buff[idxs]
            next_state_batch = self.next_state_buff[idxs]
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, state_batch, next_state_batch 

    def __len__(self):
        return self.num_stored


    def concatenate(self, state_dict):
        obs = state_dict['obs_buff']
        act = state_dict['act_buff']
        reward = state_dict['rew_buff']
        next_obs = state_dict['next_obs_buff']
        done = state_dict['done_buff']
        state, next_state = None, None
        if 'state_buff' in state_dict:
            state = state_dict['state_buff']
        if 'next_state_buff' in state_dict:
            next_state = state_dict['next_state_buff']

        self.add(obs, act, reward, next_obs, done, state, next_state)

    
    def save(self, filepath):
        state = self.get_as_tensor_dict()
        # state = {
        #     'obs_buff': self.obs_buff,
        #     'act_buff': self.act_buff,
        #     'rew_buff': self.rew_buff,
        #     'next_obs_buff': self.next_obs_buff,
        #     'done_buff': self.done_buff,
        #     'curr_idx': self.curr_idx,
        #     'num_stored': self.num_stored
        # }
        # if self.state_buff is not None:
        #     state['state_buff'] = self.state_buff
        #     state['next_state_buff'] = self.next_state_buff
        
        torch.save(state, filepath)
    
    def load(self, filepath):
        state = torch.load(filepath)
        self.obs_buff = state['obs_buff'].to(self.device)
        self.act_buff = state['act_buff'].to(self.device)
        self.rew_buff = state['rew_buff'].to(self.device)
        self.next_obs_buff = state['next_obs_buff'].to(self.device)
        self.done_buff = state['done_buff'].to(self.device)
        self.curr_idx = state['curr_idx'].to(self.device)
        self.capacity = state['num_stored']
        self.num_stored = state['num_stored'].to(self.device)
        self.obs_dim = self.obs_buff[-1]
        self.act_dim = self.act_buff[-1]

    def get_as_tensor_dict(self):
        state = {
            'obs_buff': self.obs_buff[0:self.num_stored],
            'act_buff': self.act_buff[0:self.num_stored],
            'rew_buff': self.rew_buff[0:self.num_stored],
            'done_buff': self.done_buff[0:self.num_stored],
            'next_obs_buff': self.next_obs_buff[0:self.num_stored],
            # 'capacity': self.q_acc_cmd_buff[0:self.num_stored],
            'curr_idx': self.curr_idx,
            'num_stored': self.num_stored
        }
        if self.state_buff is not None:
            state['state_buff'] = self.state_buff
            state['next_state_buff'] = self.next_state_buff
        return state