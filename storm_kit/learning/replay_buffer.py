from typing import Optional, Dict
from collections import defaultdict
import torch

class ReplayBuffer():
    def __init__(self, capacity:int, device:torch.device = torch.device('cpu')):
        self.capacity = capacity
        self.device = device
        
        self.buffers = {}        
        self.curr_idx = 0
        self.num_stored = 0

    def allocate_new_buffer(self, key, dim):
        if dim == 1:
            self.buffers[key] = torch.empty(self.capacity, device=self.device)
        else:
            self.buffers[key] = torch.empty(self.capacity, dim, device=self.device)

    def add(self, data_dict: Dict[str, torch.Tensor], ignores=("metadata",)):
        #Add a single data point to the replay buffer
        for k, v in data_dict.items():
            if not any([ig in k for ig in ignores]):
                if k not in self.buffers:
                    dim = v.shape[0]
                    self.allocate_new_buffer(k, dim)
                
                v = torch.as_tensor(v).to(self.device)
                self.buffers[k][self.curr_idx] = v
        self.curr_idx = (self.curr_idx + 1) % self.capacity
        self.num_stored = min(self.num_stored + 1, self.capacity)

    def add_batch(self, batch_data_dict: Dict[str, torch.Tensor], ignores=("metadata",)):
        #Add a batch of datapoints to the replay buffer
        for k,v in batch_data_dict.items():
            if not any([ig in k for ig in ignores]):
                if k not in self.buffers:
                    dim = 1
                    if v.ndim == 2:
                        dim = v.shape[1]
                    self.allocate_new_buffer(k, dim)
                v = torch.as_tensor(v).to(self.device)
                # if v.ndim == 1:
                #     v = v.unsqueeze(1)
                num_points = v.shape[0]
                remaining = min(self.capacity - self.curr_idx, num_points)
                if num_points > remaining:
                    #add to front
                    extra = num_points - remaining
                    self.buffers[k][0:extra] = v[-extra:]
                #add to end
                # self.buffers[k][self.curr_idx: self.curr_idx + remaining] = v[0:remaining]
                buffer_slice_shape = self.buffers[k][self.curr_idx: self.curr_idx + remaining].shape
                if buffer_slice_shape != v[0:remaining].shape:
                    v = v.squeeze(-1)
                    # print(f"Reshaped v to: {v.shape}")
                #add to the end
                # print(f"Adding to the end: self.curr_idx={self.curr_idx}, remaining={remaining}, v[0:remaining].shape={v[0:remaining].shape}")
                self.buffers[k][self.curr_idx: self.curr_idx + remaining] = v[0:remaining]
            
        self.curr_idx = (self.curr_idx + num_points) % self.capacity
        self.num_stored = min(self.num_stored + num_points, self.capacity)


    def sample(self, batch_size):
        idxs = torch.randint(
            0, len(self), size=(batch_size,), device=self.device)
        batch = {}
        for k, v in self.buffers.items():
            batch[k] = v[idxs]
        return batch
    
    def batch_iterator(self, batch_size):
        num = len(self)
        idxs = torch.randperm(num) #for _ in range(ensemble_size)
        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)

            if (j - i) < batch_size and i != 0:
                # drop incomplete last batch
                return

            batch_size = j - i
            batch_indices = idxs[i:j]
            
            batch = {}
            for k, v in self.buffers.items():
                batch[k] = v[batch_indices]

            yield batch 

    def episode_iterator(self, ignores=("metadata",)):
        """
        Returns an iterator over episodes.
        Not that this assumes dataset has full episodes only 
        with no overlap at the beginning.
        """
        assert 'timeouts' in self.buffers and 'terminals' in self.buffers
        self['timeouts'][-1] = not self['terminals'][-1]
        episode_end_idxs = torch.logical_or(self['timeouts'], self['terminals']).nonzero().squeeze(-1) + 1

        ##################        
        # ends = (self["terminals"] + self["timeouts"]) > 0
        # ends[-1] = False  # don't need to split at the end

        # # inds = torch.arange(len(ends))[ends] + 1
        # # tmp_data = dict()
        # # for k, v in self.buffers.items():
        # #     if not any([ig in k for ig in ignores]):
        # #         secs = np.diff(np.insert(inds, (0,len(inds)),  (0,len(v)))).tolist()
        # #         tmp_data[k] = v.split(secs)
        # # traj_data = [
        # #     dict(zip(tmp_data, t)) for t in zip(*tmp_data.values())
        # # ]  # convert to list of dict
        ##############################
        start_idx = 0
        for episode_num, end_idx in enumerate(episode_end_idxs):
            episode_buffer = {}
            for k,v in self.buffers.items():
                if not any([ig in k for ig in ignores]):
                    episode_buffer[k] = v[start_idx:end_idx]

            start_idx = end_idx
            # for k in episode_buffer.keys():
            #     print(episode_num, k)
            #     assert(torch.allclose(episode_buffer[k], traj_data[episode_num][k]))

            yield episode_buffer

    def concatenate(self, new_buffer):
        buffers = new_buffer.state_dict()['buffers']
        self.add_batch(buffers)
    
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
            state['buffers'][k] = v[0:self.num_stored]
        return state

    def clear(self):
        self.buffers = {} #defaultdict(lambda:{})        
        self.curr_idx = 0
        self.num_stored = 0
    
    def to(self, device:torch.device):
        if device == self.device:
            return
        else:
            for k in self.keys():
                self.buffers[k] = self.buffers[k].to(device) 
            self.device = device


    def __len__(self):
        return self.num_stored

    def __getitem__(self, item):
        return self.buffers[item][0:self.num_stored]

    def __setitem__(self, item, value):
        if len(self) > 0:
            assert value.shape[0] == len(self), 'Supplied values must be same length as existing dataset'
        self.buffers[item] = value

    def __iter__(self):
        return iter(self.buffers)

    def keys(self):
        return self.buffers.keys()

    def items(self):
        return self.buffers.items()

    def values(self):
        return self.buffers.values()

    def __repr__(self):
        str = 'num_stored={}, capacity={}, keys = {}'.format(
            self.num_stored, self.capacity, self.buffers.keys())
        return str


def train_val_split(dataset, split_ratio):
    #split into training and validation episodes
    #TODO: Randomize the trajectories
    episode_list = [episode for episode in dataset.episode_iterator()]
    num_episodes = len(episode_list)
    print('Number of episodes = {}'.format(num_episodes))
    num_train_episodes = int(split_ratio * num_episodes)
    train_episodes = episode_list[0: num_train_episodes]
    validation_episodes = episode_list[num_train_episodes:]
    # import pdb; pdb.set_trace()
    episode_keys = list(dataset.keys())
    train_batch = {k: torch.cat([ep[k] for ep in train_episodes]) for k in episode_keys}
    validation_batch = {k: torch.cat([ep[k] for ep in validation_episodes]) for k in episode_keys}
    
    
    train_buffer = ReplayBuffer(capacity=train_batch[episode_keys[0]].shape[0], device=dataset.device)
    validation_buffer = ReplayBuffer(capacity=validation_batch[episode_keys[0]].shape[0], device=dataset.device)

    train_buffer.add_batch(train_batch)
    validation_buffer.add_batch(validation_batch)

    return train_buffer, validation_buffer

def qlearning_dataset(dataset):
    #add next_obs and next_state buffers
    num_points = len(dataset)
    assert num_points > 1
    new_dataset = ReplayBuffer(capacity=num_points, device=dataset.device)
    
    for episode in dataset.episode_iterator():
        if episode['terminals'][-1] > 0: #if last timestep is terminal, we repeat it
            for k in episode.keys():
                if k.startswith(('state', 'observations')): # or k.startswith('observations'):
                    episode[k] = torch.cat((episode[k], episode[k][-1:]), dim=0)
        
        else: # if last timestep is timeout, we ignore it
            for k in episode.keys():
                if not k.startswith(('state', 'observations')): # and (not k.startswith('observations')):
                    episode[k] = episode[k][:-1]
                episode['timeouts'][-1:] = 1
        
        episode_keys = list(episode.keys())
        for k in episode_keys:
            if k.startswith(('state', 'observations')):
                episode['next_'+k] = episode[k][1:]
                episode[k] = episode[k][0:-1]
        
        new_dataset.add_batch(episode)

    return new_dataset



def qlearning_dataset2(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        if isinstance(dataset['observations'], torch.Tensor):
            obs = dataset['observations'][i].numpy().astype(np.float32)
            new_obs = dataset['observations'][i+1].numpy().astype(np.float32)
            action = dataset['actions'][i].numpy().astype(np.float32)
            reward = dataset['rewards'][i].numpy().astype(np.float32)
            done_bool = bool(dataset['terminals'][i])
            # timeouts = bool(dataset['timeouts'][i])
        else:
            obs = dataset['observations'][i].astype(np.float32)
            new_obs = dataset['observations'][i+1].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])
            # timeouts = bool(dataset['timeouts'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }



# def add_batch(self, batch_data_dict: Dict[str, torch.Tensor]):
#     #Add a batch of datapoints to the replay buffer
#     for k,v in batch_data_dict.items():
#         if k not in self.buffers:
#             if v.ndim > 1:
#                 dim = v.shape[-1]
#                 self.buffers[k] = torch.empty(self.capacity, dim, device=self.device)
#             else:
#                 self.buffers[k] = torch.empty(self.capacity, device=self.device)

#         num_points_added = self.add_to_buffer(self.buffers[k], v)
        
#     self.curr_idx = (self.curr_idx + num_points_added) % self.capacity
#     self.num_stored = min(self.num_stored + num_points_added, self.capacity)



if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    buff = ReplayBuffer(capacity=10)
    #test adding sequential data points
    for i in range(10):
        data_dict = {
            'state': torch.randn(10),
            'cost' : torch.randn(1)
        }
        buff.add(data_dict)
        print(buff['state'])
        print(len(buff))
        input('....')
    
    #now add starting from front
    for i in range(10):
        data_dict = {
            'state': i * torch.ones(10),
            'cost' : i * torch.ones(1)
        }
        buff.add(data_dict)
        print(buff['state'])
        print(len(buff))
        input('...')
    
    #test adding batch of datapoints
    buff = ReplayBuffer(capacity=10)
    batch_size=5
    data_dict = {
        'state': torch.randn(5,10),
        'cost' : torch.randn(5,)
    }
    buff.add_batch(data_dict)
    print(buff['state'].shape)
    print(buff['cost'].shape)
    print(len(buff))
    data_dict = {
        'state': torch.zeros(5,10),
        'cost' : torch.zeros(5,)
    }

    #Add zeros to back
    buff.add_batch(data_dict)
    print(buff['state'])
    print(buff['cost'])
    print(len(buff))
    #Should add zeros again to front
    buff.add_batch(data_dict)
    print(buff['state'])
    print(buff['cost'])
    print(len(buff))





# class ReplayBuffer():
#     def __init__(self, capacity:int, device:torch.device = torch.device('cpu')):
#         self.capacity = capacity
#         self.device = device
        
#         self.buffers = defaultdict(lambda:{})        
#         self.curr_idx = 0
#         self.num_stored = 0

#     def add_to_buffer(self, buff:Dict[str, torch.Tensor], v:torch.Tensor):
        
#         v = v.to(self.device)
#         num_points = v.shape[0]
#         remaining = min(self.capacity - self.curr_idx, num_points)
#         if num_points > remaining:
#             #add to front
#             extra = num_points - remaining
#             buff[0:extra] = v[-extra:]
#         #add to end
#         buff[self.curr_idx:self.curr_idx + remaining] = v[0:remaining]
#         return num_points

#     def add(self, 
#             batch_dict: Dict[str, torch.Tensor]):
        
#         for k,v in batch_dict.items():
#             if k not in self.buffers:
#                 if isinstance(v, dict):
#                     for k2, v2 in v.items():
#                         dim = v2.shape[-1]
#                         self.buffers[k][k2] = torch.empty(self.capacity, dim, device=self.device)
#                 else:
#                     if v.ndim > 1:
#                         dim = v.shape[-1]
#                         self.buffers[k] = torch.empty(self.capacity, dim, device=self.device)
#                     else:
#                         self.buffers[k] = torch.empty(self.capacity, device=self.device)

#             if isinstance(v, dict):
#                 for k2, v2 in v.items():
#                     num_points_added = self.add_to_buffer(self.buffers[k][k2], v2)
#             else:
#                 num_points_added = self.add_to_buffer(self.buffers[k], v)
            
#         self.curr_idx = (self.curr_idx + num_points_added) % self.capacity
#         self.num_stored = min(self.num_stored + num_points_added, self.capacity)

#     def sample(self, batch_size): #, sample_next_state:bool=False):
#         idxs = torch.randint(0, len(self), size=(batch_size,), device=self.device)
        
#         batch = defaultdict(lambda:{})
#         for k, v in self.buffers.items():
#             # name = k.split('_buff')[0]
#             if isinstance(v, dict):
#                 for k2, v2 in v.items():
#                     batch[k][k2] = v2[idxs]
#             else:
#                 batch[k] = v[idxs]
#                 # if sample_next_state and k.startswith('state'):
#                 #     batch['next_' + k] = v[idxs + 1]
#         return batch
    
#     # def qlearning_dataset(self):
#     #     #add next_state buffers
#     #     num_points = len(self)
#     #     assert num_points > 1
#     #     new_dataset = RobotBuffer(capacity=num_points-1, device=self.device)
#     #     buff_dict = {}
#     #     for k in self.buffers:
#     #         if k != 'timeouts':
#     #             # if k == 'terminals':
#     #             #     curr_buff = self.buffers[k][1:num_points]
#     #             # else:
#     #             curr_buff = self.buffers[k][0:num_points-1]
                
#     #             if k.startswith('state'):
#     #                 next_state_buff = self.buffers[k][1:num_points]
            
#     #             #now we remove the elements corresponding to timeouts
#     #             if 'timeouts' in self.buffers:
#     #                 timeouts = self.buffers['timeouts'][0:num_points-1]
#     #                 curr_buff = curr_buff[~timeouts.bool()]
#     #                 buff_dict[k] = curr_buff
#     #                 if k.startswith('state'):
#     #                     next_state_buff = next_state_buff[~timeouts.bool()]
#     #                     buff_dict['next_'+k] = next_state_buff
        
#     #     new_dataset.add(buff_dict)
#     #     # new_dataset.curr_idx = (new_dataset.curr_idx + num_points) % new_dataset.capacity
#     #     # new_dataset.num_stored = min(new_dataset.num_stored + num_points, new_dataset.capacity)
#     #     return new_dataset

#     def qlearning_dataset(self):
#         #add next_state buffers
#         num_points = len(self)
#         assert num_points > 1
#         new_dataset = RobotBuffer(capacity=num_points, device=self.device)
#         for episode in self.episode_iterator():
#             if episode['terminals'][-1] > 0:
#                 for k in episode.keys():
#                     if k.startswith('state'):
#                         episode[k] = torch.cat((episode[k], episode[k][-1:]), dim=0)
            
#             else: #timeout
#                 for k in episode.keys():
#                     if not k.startswith('state'):
#                         episode[k] = episode[k][:-1]
#                     episode['timeouts'][-1:] = 1
            
#             episode_keys = list(episode.keys())
#             for k in episode_keys:
#                 if k.startswith('state'):
#                     episode['next_'+k] = episode[k][1:]
#                     episode[k] = episode[k][0:-1]
#             new_dataset.add(episode)

#         return new_dataset

#     def episode_iterator(self, max_episode_length=None):
#         """Returns an iterator over episodes."""

#         terminals = self.buffers['terminals'][0:len(self)]
#         if 'timeouts' in self.buffers:
#             timeouts = self.buffers['timeouts'][0:len(self)]
#         else:
#             raise NotImplementedError()
#             # num_episodes = len(self) // max_episode_length
#             # timeouts = torch.linspace(0, num_episodes, num_episodes)
#             # print(num_episodes, timeouts, len(self), max_episode_length)
#             # input('....')

#         episode_end_idxs = torch.logical_or(terminals, timeouts).nonzero()

#         start_idx = 0
#         for end_idx in episode_end_idxs:
#             episode_buffer = {}
#             for k,v in self.buffers.items():
#                 episode_buffer[k] = v[start_idx:end_idx+1]

#             start_idx = end_idx + 1
#             yield episode_buffer


#     def concatenate(self, new_buffer):
#         buffers = new_buffer.state_dict()['buffers']
#         self.add(buffers)
    
#     def save(self, filepath):
#         state = self.state_dict()
#         torch.save(state, filepath)
    
#     def load(self, filepath):
#         state = torch.load(filepath)
#         self.capacity = state['num_stored']
#         self.num_stored = state['num_stored']

#         buffers = state['buffers']
        
#         for k, v in buffers.items():
#             if isinstance(v, dict):
#                 self.buffers[k] = {}
#                 for k2, v2 in v.items():
#                     self.buffers[k][k2] = v2.to(self.device)
#             else:
#                 self.buffers[k] = v.to(self.device)

#     def state_dict(self):
#         state = dict(buffers={}, curr_idx=self.curr_idx, num_stored=self.num_stored)
#         for k, v in self.buffers.items():
#             if isinstance(v, dict):
#                 state['buffers'][k]={}
#                 for k2, v2 in v.items():
#                     state['buffers'][k][k2] = v2[0:self.num_stored]
#             else:
#                 state['buffers'][k] = v[0:self.num_stored]
#         return state

#     def __len__(self):
#         return self.num_stored

#     def __repr__(self):
#         str = 'num_stored={}, capacity={}, keys = {}'.format(
#             self.num_stored, self.capacity, self.buffers.keys())
#         return str

#     def __getitem__(self, item):
#         return self.buffers[item]

#     def __setitem__(self, item, value):
#         self.buffers[item] = value