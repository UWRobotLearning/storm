from typing import Optional
import copy
import os
import torch
import torch.nn as nn


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
        # self.obs_dict = None
        # self.state_dict = None
        # self.targets = None
        self.log_freq = self.cfg.get('log_freq', -1)
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

    def collect_experience(self, num_episodes:int, update_buffer:bool=True, deterministic:bool=True, debug:bool=False):
        
        # if self.train_rng is not None:
        #     train_rng_state_before = self.train_rng.get_state()

        buff = None
        if update_buffer:
            buff = self.buffer
        
        buff, metrics = self.runner_fn(
            envs = self.envs,
            num_episodes = num_episodes, 
            policy = self.policy,
            task = self.task,
            buffer = buff,
            deterministic = deterministic,
            debug = debug,
            device = self.device,
            rng=self.train_rng
        )

        # if self.eval_rng is not None:
        #     self.eval_rng.set_state(eval_rng_state_before)


        return buff, metrics

    def evaluate_policy(self, policy, num_eval_episodes:int, deterministic:bool=True, debug:bool=False):

        if self.eval_rng is not None:
            eval_rng_state_before = self.eval_rng.get_state()
        
        _, play_metrics = self.runner_fn(
            envs=self.envs,
            num_episodes=num_eval_episodes,
            policy=policy,
            task=self.task,
            buffer=None,
            deterministic=deterministic,
            debug=debug,
            device=self.device,
            rng=self.eval_rng
        )

        if self.eval_rng is not None:
            self.eval_rng.set_state(eval_rng_state_before)
        
        return play_metrics

    def relabel_batch(self, batch_dict, relabel_cost_and_term=False):

        act_batch = batch_dict['actions']
        state_batch = batch_dict['state_dict']
        next_obs_batch = batch_dict['next_obs']
        next_state_batch = batch_dict['next_state_dict']
        done_batch = batch_dict['done'].squeeze(-1).float()
        goal_dict = batch_dict['goal_dict']

        if 'filtered_state_dict' in batch_dict:
            state_batch = batch_dict['filtered_state_dict']

        #TODO: Call task.update_params here to make sure goal is updated
        self.task.update_params(dict(goal_dict=goal_dict))
        new_obs = self.task.compute_observations(state_batch, compute_full_state=True, debug=True)
        # new_obs, new_cost, new_termination, _, _ = self.task.forward(state_batch, act_batch)
        batch_dict['obs'] = new_obs.clone()

        # new_next_obs = self.task.compute_observations(next_state_batch, compute_full_state=True)

        # batch_dict['next_obs'] = new_next_obs.clone()



        return batch_dict

