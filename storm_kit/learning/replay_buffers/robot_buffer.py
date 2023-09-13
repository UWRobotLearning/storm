from typing import Optional, Dict
from collections import defaultdict
from storm_kit.learning.replay_buffers import ReplayBuffer
import torch

class RobotBuffer(ReplayBuffer):
    def __init__(self, capacity:int, device:torch.device = torch.device('cpu')):
        self.capacity = capacity
        self.device = device
        
        self.buffers = defaultdict(lambda:{})        
        self.curr_idx = 0
        self.num_stored = 0

    def add_to_buffer(self, buff:Dict[str, torch.Tensor], v:torch.Tensor):

        v = v.to(self.device)
        num_points = v.shape[0]
        remaining = min(self.capacity - self.curr_idx, num_points)
        if num_points > remaining:
            #add to front
            extra = num_points - remaining
            buff[0:extra] = v[-extra:]
        #add to end
        buff[self.curr_idx:self.curr_idx + remaining] = v[0:remaining]
        return num_points


    def add(self, 
            batch_dict: Dict[str, torch.Tensor]):
        
        for k,v in batch_dict.items():
            if k not in self.buffers:
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        dim = v2.shape[-1]
                        self.buffers[k][k2] = torch.empty(self.capacity, dim, device=self.device)
                else:
                    if v.ndim > 1:
                        dim = v.shape[-1]
                        self.buffers[k] = torch.empty(self.capacity, dim, device=self.device)
                    else:
                        self.buffers[k] = torch.empty(self.capacity, device=self.device)

            if isinstance(v, dict):
                for k2, v2 in v.items():
                    num_points_added = self.add_to_buffer(self.buffers[k][k2], v2)
            else:
                num_points_added = self.add_to_buffer(self.buffers[k], v)
            
        self.curr_idx = (self.curr_idx + num_points_added) % self.capacity
        self.num_stored = min(self.num_stored + num_points_added, self.capacity)

    def sample(self, batch_size):
        idxs = torch.randint(0, len(self), size=(batch_size,), device=self.device)
        batch = defaultdict(lambda:{})
        for k, v in self.buffers.items():
            # name = k.split('_buff')[0]
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    batch[k][k2] = v2[idxs]
            else:
                batch[k] = v[idxs] #self.buffers[k][idxs]
        return batch

    def concatenate(self, state_dict):
        buffers = state_dict['buffers']
        self.add(buffers)
        # batch_dict = {}
        # for k, v in buffers:
        #     # name, _ = k.split("_")
        #     if isinstance(v, dict):

        #     else:
        #         batch_dict[name] = v
        # self.add(batch_dict)
    
    def save(self, filepath):
        state = self.state_dict()
        torch.save(state, filepath)
    
    def load(self, filepath):
        state = torch.load(filepath)
        self.capacity = state['num_stored']
        self.num_stored = state['num_stored']

        buffers = state[buffers]
        for k, v in buffers:
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    self.buffers[k][k2] = v2.to(self.device)
            else:
                self.buffers[k] = v.to(self.device)

    def state_dict(self):
        state = {}
        for k, v in self.buffers.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    state['buffers'][k][k2] = v2[0:self.num_stored]
            else:
                state['buffers'][k] = v[0:self.num_stored]
        state['curr_idx'] = self.curr_idx
        state['num_stored'] = self.num_stored
        return state

    def __len__(self):
        return self.num_stored

    def __repr__(self):
        str = 'num_stored={}, capacity={}, keys = {}'.format(
            self.num_stored, self.capacity, self.buffers.keys())
        return str
