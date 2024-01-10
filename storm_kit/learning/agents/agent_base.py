from typing import Optional, Dict
import copy
import os
import torch
import torch.nn as nn
from storm_kit.learning.learning_utils import plot_episode


class Agent(nn.Module):
    def __init__(
        self,
        cfg,
        envs,
        task,
        obs_dim, 
        action_dim,
        buffer,
        policy,
        runner_fn,
        logger=None,
        tb_writer=None,
        device=torch.device('cpu'),
        train_rng:Optional[torch.Generator]=None,
        eval_rng:Optional[torch.Generator]=None
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.envs = envs
        self.task = task
        self.policy = policy
        self.runner_fn = runner_fn
        self.buffer = buffer
        self.device = device
        self.logger = logger
        self.tb_writer = tb_writer
        self.train_rng = train_rng
        self.eval_rng = eval_rng
        # self.log_freq = self.cfg.get('log_freq', -1)
        self.eval_freq = self.cfg.get('eval_freq', -1)  
        self.checkpoint_freq = self.cfg.get('checkpoint_freq', -1)
        self.relabel_data = self.cfg.get('relabel_data', False)
        self.init_buffers()


    def init_buffers(self):
        self.curr_rewards = torch.zeros(self.envs.num_envs, device=self.device)
        self.episode_lens = torch.zeros(self.envs.num_envs, device=self.device)
        self.episode_reward_buffer = []
        self.curr_idx = 0
        self.total_episodes_done = 0
        self.avg_episode_reward = 0.0

    def update(self):
        return {}

    def train(self, model_dir:Optional[str]=None):
        pass

    def save(self, model_dir:Optional[str]=None, data_dir:Optional[str]=None, iter:int=0):
        
        if model_dir is not None:
            print('Saving agent models and state = {}'.format(len(self.buffer)))
            state = {
                'iter': iter,
                'policy_state_dict': self.policy.state_dict(),
            }
            torch.save(state, os.path.join(model_dir, 'agent_checkpoint_{}.pt'.format(iter)))
        if data_dir is not None:
            print('Saving buffer len = {}'.format(len(self.buffer)))
            self.buffer.save(os.path.join(data_dir, 'agent_buffer_{}.pt'.format(iter)))

    def load(self, checkpoint_path:str, buffer_path:Optional[str]=None):
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if buffer_path is not None:
            self.buffer.load(buffer_path)

    def collect_experience(self, policy, num_episodes:int, update_buffer:bool=True, deterministic:bool=True, debug:bool=False):
        
        # if self.train_rng is not None:
        #     train_rng_state_before = self.train_rng.get_state()
        
        buff, metrics = self.runner_fn(
            envs = self.envs,
            num_episodes = num_episodes, 
            policy = policy,
            task = self.task,
            # buffer = buff,
            collect_data = update_buffer,
            deterministic = deterministic,
            debug = debug,
            device = self.device,
            rng=self.train_rng
        )

        if debug:
            for episode in buff.episode_iterator():
                plot_episode(episode, block=False)

        # if self.eval_rng is not None:
        #     self.eval_rng.set_state(eval_rng_state_before)


        return buff, metrics

    def evaluate_policy(self, policy, num_eval_episodes:int, deterministic:bool=True, debug:bool=False):

        if self.eval_rng is not None:
            eval_rng_state_before = self.eval_rng.get_state()
        
        eval_buffer, eval_metrics = self.runner_fn(
            envs=self.envs,
            num_episodes=num_eval_episodes,
            policy=policy,
            task=self.task,
            collect_data=debug,
            deterministic=deterministic,
            debug=debug,
            device=self.device,
            rng=self.eval_rng
        )

        if self.eval_rng is not None:
            self.eval_rng.set_state(eval_rng_state_before)
        
        if debug:
            for episode in eval_buffer.episode_iterator():
                plot_episode(episode, block=False)
        
        return eval_buffer, eval_metrics

    def preprocess_batch(self, batch_dict:Dict[str, torch.Tensor], compute_cost_and_terminals:bool=False):

        # act_batch = batch_dict['actions']
        # state_batch = batch_dict['state_dict']
        # next_obs_batch = batch_dict['next_obs']
        # next_state_batch = batch_dict['next_state_dict']
        # done_batch = batch_dict['done'].squeeze(-1).float()
        # goal_dict = batch_dict['goal_dict']

        # if 'filtered_state_dict' in batch_dict:
        #     state_batch = batch_dict['filtered_state_dict']
        new_batch_dict = {}
        state_dict = {}
        state_dict_filt = {}
        next_state_dict = {}
        next_state_dict_filt = {}
        goal_dict = {}

        #TODO: This part of the code can be made more readable by splitting the keys
        for k in batch_dict:
            value = batch_dict[k]
            if k.startswith('states'):
                state_dict[k[len('states/'):]] = value
            elif k.startswith('next_states'):
                next_state_dict[k[len('next_states/'):]] = value
            elif k.startswith('goal'):
                goal_dict[k[len('goal/'):]] = value
            else:
                new_batch_dict[k] = value

        for k in state_dict.keys():
            if k.startswith('filtered'):
                state_dict_filt[k[len('filtered/'):]] = state_dict[k]

        for k in next_state_dict.keys():
            if k.startswith('filtered'):
                next_state_dict_filt[k[len('filtered/'):]] = next_state_dict[k]

        # new_batch_dict['state_dict'] = state_dict_filt if len(state_dict_filt.keys()) > 0 else state_dict        
        # new_batch_dict['next_state_dict'] = next_state_dict_filt if len(next_state_dict_filt.keys()) > 0 else next_state_dict
        new_batch_dict['state_dict'] =  copy.deepcopy(state_dict)        
        new_batch_dict['next_state_dict'] = copy.deepcopy(next_state_dict)
        new_batch_dict['goal_dict'] = copy.deepcopy(goal_dict)

        self.task.update_params(dict(goal_dict=new_batch_dict['goal_dict']))
        with torch.no_grad():
            full_state_dict = self.task._compute_full_state(new_batch_dict['state_dict'])
            
            cost_terms = None
            if compute_cost_and_terminals:
                cost, cost_terms = self.task.compute_cost(full_state_dict)
                # term, term_cost, term_info = self.task.compute_termination(
                #     new_batch_dict['next_state_dict'], compute_full_state=True)
                terminals, term_cost, term_info = self.task.compute_termination(
                    state_dict, compute_full_state=True,
                )
                cost += term_cost 
                new_batch_dict['cost'] = cost.clone()
                cost_terms = {**cost_terms, **term_info}
                # print(new_batch_dict['terminals'])
                # print(terminals)
                # print(term_cost)
                # assert torch.allclose(terminals, new_batch_dict['terminals'])
                # input('...........................====')

            obs = self.task.compute_observations(
               full_state_dict , compute_full_state=False, cost_terms=cost_terms)
            new_batch_dict['obs'] = obs.clone()


        if len(new_batch_dict['next_state_dict'].keys()) > 0:
            next_obs = self.task.compute_observations(new_batch_dict['next_state_dict'], compute_full_state=True)
            new_batch_dict['next_obs'] = next_obs.clone()

        return new_batch_dict

