import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from storm_kit.learning.agents import Agent


class SACAgent(Agent):
    def __init__(
            self,
            cfg,
            envs,
            task,
            obs_dim, 
            action_dim,
            buffer,
            policy,
            critic,
            runner_fn,
            logger=None,
            tb_writer=None,
            device=torch.device('cpu'),
    ):


        super().__init__(
            cfg, envs, task, obs_dim, action_dim,
            buffer=buffer, policy=policy,
            runner_fn=runner_fn, logger=logger, 
            tb_writer=tb_writer, device=device        
        )
        self.critic = critic
        self.target_critic = copy.deepcopy(self.critic)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), 
                                    lr=self.cfg['policy_optimizer']['lr'])
        self.critic_optimizer =  optim.Adam(self.critic.parameters(), 
                                    lr=self.cfg['critic_optimizer']['lr'])
        self.polyak_tau = self.cfg['polyak_tau']
        self.discount = self.cfg['discount']
        # self.updates_per_train_step = self.cfg['updates_per_train_step']
        self.num_action_samples = self.cfg['num_action_samples']
        self.num_train_episodes = self.cfg['num_train_episodes']
        self.num_update_steps = self.cfg['num_update_steps']
        self.automatic_entropy_tuning = self.cfg['automatic_entropy_tuning']

        if self.automatic_entropy_tuning:
            self.log_alpha = nn.Parameter(torch.tensor(self.cfg['init_log_alpha']))
            self.target_entropy = -np.prod(self.action_dim)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.cfg['alpha_optimizer']['lr'])
        else:
            self.alpha = torch.tensor(self.cfg['fixed_alpha'])
            self.log_alpha = torch.log(self.alpha)


    def train(self, model_dir=None):
        self.obs_dict = self.envs.reset()
        num_train_steps = self.cfg['num_train_steps']
        total_env_steps = 0

        # self.best_policy = copy.deepcopy(self.policy)
        # best_policy_perf = -torch.inf
        # best_policy_step = 0
        
        pbar = tqdm(range(int(num_train_steps)), desc='train')
        for i in pbar:
            # step_start_time = time.time()
            #collect new experience
            # play_metrics = self.collect_experience(num_steps_per_env=self.num_steps_per_env, update_buffer=True)
            self.buffer, play_metrics = self.runner_fn(
                envs=self.envs,
                num_episodes=self.num_train_episodes, 
                policy=self.policy,
                task=self.task,
                buffer=self.buffer,
                device=self.device
            )

            total_env_steps += play_metrics['num_steps_collected']
            #update agent
            for _ in range(self.num_update_steps):
                batch = self.buffer.sample(self.cfg['train_batch_size'])
                # update_start_time = time.time()
                train_metrics = self.update(*batch)
                # update_time = time.time() - update_start_time
                pbar.set_postfix(train_metrics)

            if (i % self.log_freq == 0) or (i == num_train_steps -1):
                if self.tb_writer is not None:
                    for k, v in play_metrics.items():
                        self.tb_writer.add_scalar('Train/' + k, v, total_env_steps)
                    for k, v in train_metrics.items():
                        self.tb_writer.add_scalar('Train/' + k, v, total_env_steps)
                        
            # eval_metrics = {}
            # if (i % self.eval_freq == 0) or (i == num_train_steps -1):
            #     # eval_start_time = time.time()
            #     eval_metrics = self.evaluate(num_eval_episodes= self.cfg['num_eval_episodes'])
            #     if self.logger is not None:
            #         self.logger.row(eval_metrics, nostdout=True)
            #     if self.tb_writer is not None:
            #         for k, v in eval_metrics.items():
            #             self.tb_writer.add_scalar('Eval/' + k, v, i)

            #     if eval_metrics['eval_episode_reward_avg'] >= best_policy_perf:
            #         self.best_policy = copy.deepcopy(self.policy)
            #         best_policy_perf = eval_metrics['eval_episode_reward_avg']
            #         best_policy_step = i

            #     self.policy.train()
            #     pbar.set_postfix(eval_metrics)

    def update(self, obs_batch, act_batch, cost_batch, next_obs_batch, done_batch, state_batch=None, next_state_batch=None):
        obs_batch = obs_batch.to(self.device)
        act_batch = act_batch.to(self.device)
        cost_batch = cost_batch.to(self.device)
        next_obs_batch = next_obs_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        #Update critic
        self.critic_optimizer.zero_grad()
        critic_loss, avg_q_value = self.compute_critic_loss(obs_batch, act_batch, cost_batch, next_obs_batch, done_batch)
        critic_loss.backward()
        self.critic_optimizer.step()

        #Update target critic using exponential moving average
        for (param1, param2) in zip(self.target_critic.parameters(), self.critic.parameters()):
            param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)

        #Update policy
        self.policy_optimizer.zero_grad()
        policy_loss, log_pi_new_actions = self.compute_policy_loss(obs_batch)
        policy_loss.backward()
        self.policy_optimizer.step()

        #Update temperature
        alpha_loss = torch.tensor([0.0])
        if self.automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss = self.log_alpha * (-log_pi_new_actions.detach() - self.target_entropy)
            alpha_loss.backward()
            self.alpha_optimizer.step()

            
        train_metrics = {}
        train_metrics['critic_loss'] = critic_loss.item()
        train_metrics['policy_loss'] = policy_loss.item()
        train_metrics['policy_entropy'] = log_pi_new_actions.item()
        train_metrics['avg_q_value'] = avg_q_value.item()
        train_metrics['alpha_loss'] = alpha_loss.item()
        train_metrics['alpha'] = torch.exp(self.log_alpha).item()
        return train_metrics


    def compute_policy_loss(self, obs_batch):
        new_action_dist = self.policy({'obs': obs_batch})
        new_actions = new_action_dist.rsample(torch.Size([self.num_action_samples]))
        log_pi_new_actions = new_action_dist.log_prob(new_actions).sum(-1).mean(0)
        q_pred = self.critic(
            {'obs': obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
            new_actions).mean(0)
        
        alpha = torch.exp(self.log_alpha).item()
        policy_loss = (alpha * log_pi_new_actions - q_pred).mean() 

        return policy_loss, log_pi_new_actions.mean()


    def compute_critic_loss(self, obs_batch, act_batch, cost_batch, next_obs_batch, done_batch):

        # def compute_bellman_target(q_pred_next):
        #     assert cost_batch.shape == q_pred_next.shape
        #     return (cost_batch + (1.-done_batch.float())*self.discount*q_pred_next)#.clamp(min=self._Vmin, max=self._Vmax)
        rew_batch = -1.0 * cost_batch #TODO: fix
        with torch.no_grad():
            next_actions_dist = self.policy({'obs': next_obs_batch})
            next_actions = next_actions_dist.rsample(torch.Size([self.num_action_samples]))
            next_actions_log_prob = next_actions_dist.log_prob(next_actions).sum(-1).mean(0)
            
            target_pred = self.target_critic(
                {'obs': next_obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
                next_actions).mean(0)
            
            alpha = torch.exp(self.log_alpha).item()
            q_pred_next =  target_pred - alpha * next_actions_log_prob
            q_target = rew_batch +  (1. - done_batch.float()) * self.discount * q_pred_next

        qf1, qf2 = self.critic.both({'obs': obs_batch}, act_batch)
        qf1_loss = F.mse_loss(qf1,  q_target, reduction='none').mean()
        qf2_loss = F.mse_loss(qf2, q_target, reduction='none').mean()
        qf_loss = qf1_loss + qf2_loss

        avg_q_value = torch.min(qf1, qf2).mean()
        
        return qf_loss, avg_q_value


        