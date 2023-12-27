from typing import Optional, Dict
from collections import defaultdict
import torch

class RobotBuffer():
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

    def sample(self, batch_size, sample_next_state:bool=False):
        idxs = torch.randint(0, len(self), size=(batch_size,), device=self.device)
        
        batch = defaultdict(lambda:{})
        for k, v in self.buffers.items():
            # name = k.split('_buff')[0]
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    batch[k][k2] = v2[idxs]
            else:
                batch[k] = v[idxs]
                if sample_next_state and k.startswith('state'):
                    batch['next_' + k] = v[idxs + 1]
        return batch
    
    def qlearning_dataset(self):
        #add next_state buffers
        num_points = len(self)
        assert num_points > 1
        new_dataset = RobotBuffer(capacity=num_points-1, device=self.device)
        buff_dict = {}
        for k in self.buffers:
            if k != 'timeouts':
                # if k == 'terminals':
                #     curr_buff = self.buffers[k][1:num_points]
                # else:
                curr_buff = self.buffers[k][0:num_points-1]
                
                if k.startswith('state'):
                    next_state_buff = self.buffers[k][1:num_points]
            
                #now we remove the elements corresponding to timeouts
                if 'timeouts' in self.buffers:
                    timeouts = self.buffers['timeouts'][0:num_points-1]
                    curr_buff = curr_buff[~timeouts.bool()]
                    buff_dict[k] = curr_buff
                    if k.startswith('state'):
                        next_state_buff = next_state_buff[~timeouts.bool()]
                        buff_dict['next_'+k] = next_state_buff
        
        new_dataset.add(buff_dict)
        # new_dataset.curr_idx = (new_dataset.curr_idx + num_points) % new_dataset.capacity
        # new_dataset.num_stored = min(new_dataset.num_stored + num_points, new_dataset.capacity)
        return new_dataset

    def episode_iterator(self, max_episode_length=None):
        """Returns an iterator over episodes."""

        terminals = self.buffers['terminals'][0:len(self)]
        if 'timeouts' in self.buffers:
            timeouts = self.buffers['timeouts'][0:len(self)]
        else:
            raise NotImplementedError()
            # num_episodes = len(self) // max_episode_length
            # timeouts = torch.linspace(0, num_episodes, num_episodes)
            # print(num_episodes, timeouts, len(self), max_episode_length)
            # input('....')

        episode_end_idxs = torch.logical_or(terminals, timeouts).nonzero()

        start_idx = 0
        for end_idx in episode_end_idxs:
            episode_buffer = {}
            for k,v in self.buffers.items():
                episode_buffer[k] = v[start_idx:end_idx+1]

            start_idx = end_idx + 1
            yield episode_buffer


    def concatenate(self, new_buffer):
        buffers = new_buffer.state_dict()['buffers']
        self.add(buffers)
    
    def save(self, filepath):
        state = self.state_dict()
        torch.save(state, filepath)
    
    def load(self, filepath):
        state = torch.load(filepath)
        self.capacity = state['num_stored']
        self.num_stored = state['num_stored']

        buffers = state['buffers']
        
        for k, v in buffers.items():
            if isinstance(v, dict):
                self.buffers[k] = {}
                for k2, v2 in v.items():
                    self.buffers[k][k2] = v2.to(self.device)
            else:
                self.buffers[k] = v.to(self.device)

    def state_dict(self):
        state = dict(buffers={}, curr_idx=self.curr_idx, num_stored=self.num_stored)
        for k, v in self.buffers.items():
            if isinstance(v, dict):
                state['buffers'][k]={}
                for k2, v2 in v.items():
                    state['buffers'][k][k2] = v2[0:self.num_stored]
            else:
                state['buffers'][k] = v[0:self.num_stored]
        return state

    def __len__(self):
        return self.num_stored

    def __repr__(self):
        str = 'num_stored={}, capacity={}, keys = {}'.format(
            self.num_stored, self.capacity, self.buffers.keys())
        return str

    def __getitem__(self, item):
        return self.buffers[item]

    def __setitem__(self, item, value):
        self.buffers[item] = value