import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from storm_kit.learning.agents import Agent
from storm_kit.learning.learning_utils import logmeanexp


class MPOAgent(Agent):
    def __init__(
            self,
            cfg,
            envs,
            obs_space, 
            action_space,
            buffer,
            policy,
            critic,
            logger=None,
            tb_writer=None,
            device=torch.device('cpu'),
    ):


        super().__init__(
            cfg, envs, obs_space, action_space,
            buffer=buffer, policy=policy,
            logger=logger, tb_writer=tb_writer,
            device=device        
        )
        self.critic = critic
        self.target_critic = copy.deepcopy(self.critic)
        self.target_policy = copy.deepcopy(self.policy)

        for param in self.target_critic.parameters():
            param.requires_grad_(False)

        for param in self.target_policy.parameters():
            param.requires_grad_(False)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), 
                                    lr=self.cfg['policy_optimizer']['lr'])
        self.critic_optimizer =  optim.Adam(self.critic.parameters(), 
                                    lr=self.cfg['critic_optimizer']['lr'])
        self.polyak_tau = self.cfg['polyak_tau']
        self.discount = self.cfg['discount']
        # self.updates_per_train_step = self.cfg['updates_per_train_step']
        self.num_steps_per_env = self.cfg['num_steps_per_env']
        self.num_update_steps = self.cfg['num_update_steps']
        self.num_action_samples = self.cfg['num_action_samples']
        self.num_m_step_iterations = self.cfg['num_m_step_iterations']
        self.q_transform = self.cfg['q_transform']
        self.target_critic_update_freq = self.cfg['target_critic_update_freq']
        # self.automatic_temperature_tuning = self.cfg['automatic_temperature_tuning']
        self.epsilon = self.cfg['epsilon']
        self.epsilon_kl = self.cfg['epsilon_kl']
        self.init_eta = self.cfg['init_eta']
        self.init_alpha = self.cfg['init_alpha']
        self.num_eta_iterations = self.cfg['num_eta_iterations']
        self.max_alpha = self.cfg['max_alpha']

        # if self.automatic_temperature_tuning:
        self.eta = nn.Parameter(torch.tensor(self.init_eta, device=self.device))
        self.eta_optimizer = optim.Adam([self.eta], lr=self.cfg['eta_optimizer']['lr'])
        self.alpha = nn.Parameter(torch.tensor(self.init_alpha, device=self.device))
        self.alpha_optimizer = optim.Adam([self.alpha], lr=self.cfg['alpha_optimizer']['lr'])

    def train(self, model_dir=None):
        self.obs_dict = self.envs.reset()
        num_train_steps = self.cfg['num_train_steps']
        total_env_steps = 0
        total_update_steps = 0

        # self.best_policy = copy.deepcopy(self.policy)
        # best_policy_perf = -torch.inf
        # best_policy_step = 0
        
        pbar = tqdm(range(int(num_train_steps)), desc='train')
        for i in pbar:
            #collect new experience
            play_metrics = self.collect_experience(
                num_steps_per_env=self.num_steps_per_env, 
                update_buffer=True,
                policy=self.target_policy)
            total_env_steps += play_metrics['num_steps_collected']
            #update agent
            # print('params before')
            # for (param1, param2) in zip(self.target_policy.parameters(), self.policy.parameters()):
            #     print(param1, param2)
            # out('....')

            for _ in range(self.num_update_steps):
                batch = self.buffer.sample(self.cfg['train_batch_size'])
                train_metrics = self.update(*batch)
                total_update_steps += 1
                # Update target critic using exponential moving average
                for (param1, param2) in zip(self.target_critic.parameters(), self.critic.parameters()):
                    param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)




            # Update target policy  
            for (param1, param2) in zip(self.target_policy.parameters(), self.policy.parameters()):
                param1.data.copy_(param2.data)            
                

            # if total_update_steps >= self.target_critic_update_freq:
            #     for (param1, param2) in zip(self.target_critic.parameters(), self.critic.parameters()):
            #         param1.data.copy_(param2.data)
            #     total_update_steps = 0

            postfix_metrics = {}
            for k, v in train_metrics.items():
                postfix_metrics[k] = v
            for k,v in play_metrics.items():
                postfix_metrics[k] = v
            pbar.set_postfix(postfix_metrics)

            if (i % self.log_freq == 0) or (i == num_train_steps -1):
                if self.tb_writer is not None:
                    for k, v in play_metrics.items():
                        self.tb_writer.add_scalar('Train/' + k, v, total_env_steps)
                    for k, v in train_metrics.items():
                        self.tb_writer.add_scalar('Train/' + k, v, total_env_steps)
                        
    def update(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, state_batch=None, next_state_batch=None):
        obs_batch = obs_batch.to(self.device)
        act_batch = act_batch.to(self.device)
        rew_batch = rew_batch.to(self.device)
        next_obs_batch = next_obs_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        #Update critic
        critic_loss, avg_q_value, avg_target_q_value = self.update_critic(obs_batch, act_batch, rew_batch, next_obs_batch, done_batch)

        #Update policy
        policy_loss, eta_loss, alpha_grad, log_pi_new_actions = self.update_policy(obs_batch)
           
        train_metrics = {}
        train_metrics['critic_loss'] = critic_loss
        train_metrics['policy_loss'] = policy_loss
        train_metrics['policy_entropy'] = log_pi_new_actions
        train_metrics['avg_q_value'] = avg_q_value
        train_metrics['avg_target_q_value'] = avg_target_q_value
        train_metrics['eta_loss'] = eta_loss
        train_metrics['alpha_grad'] = alpha_grad
        train_metrics['eta'] = self.eta.item()
        train_metrics['alpha'] = self.alpha.item()
        return train_metrics


    def update_policy(self, obs_batch):
        with torch.no_grad():
            old_policy_action_dist = self.target_policy({'obs': obs_batch})
            old_policy_actions = old_policy_action_dist.rsample(torch.Size([self.num_action_samples]))
            log_pi_actions_old = old_policy_action_dist.log_prob(old_policy_actions).sum(-1)
            q_pred_target = self.target_critic(
                {'obs': obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
                old_policy_actions)

            #compute updated sample-based distribution
            # with torch.no_grad():
            weights = self.compute_action_weights(old_policy_actions, q_pred_target)

        #compute temperature loss dual function optimization
        for _ in range(self.num_eta_iterations):
            self.eta_optimizer.zero_grad()
            # eta_loss = self.eta * self.epsilon + self.eta * torch.logsumexp((1.0/self.eta) * q_pred_target, dim=0).mean()
            eta_loss = self.eta * self.epsilon + self.eta * torch.mean(logmeanexp((1.0/self.eta) * q_pred_target, dim=0))
            eta_loss.backward()
            self.eta_optimizer.step()
            self.eta.data = torch.clamp(self.eta.data, min=1e-6)

        for _ in range(self.num_m_step_iterations):
            new_policy_action_dist = self.policy({'obs': obs_batch})
            log_pi_actions_new = new_policy_action_dist.log_prob(old_policy_actions).sum(-1)
            kl_div = (log_pi_actions_old - log_pi_actions_new).sum(0).mean()
            #update lagrange multiplier
            self.alpha_optimizer.zero_grad()
            # alpha_loss = self.alpha * (self.epsilon_kl - kl_div.detach())
            alpha_grad = (self.epsilon_kl - kl_div).detach()
            self.alpha.grad = alpha_grad
            self.alpha_optimizer.step()
            self.alpha.data = torch.clip(self.alpha.data, min = 0.0, max=self.max_alpha)
            #compute policy loss for improved parametric policy
            nll_loss = (weights * log_pi_actions_new).sum(0).mean()
            kl_loss = self.alpha.detach() * (self.epsilon_kl - kl_div)
            policy_loss = - (nll_loss + kl_loss)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(self.policy.parameters(), 0.1)
            self.policy_optimizer.step()

        # policy_loss = (alpha * log_pi_actions_new - q_pred).mean() 

        return policy_loss.item(), eta_loss.item(), alpha_grad.item(), log_pi_actions_new.mean().item()


    def compute_action_weights(self, actions:torch.Tensor, q_vals: torch.Tensor):
        if self.q_transform == 'exponential':
            eta = self.eta
            weights = torch.softmax((1.0 / eta) * q_vals, dim=0)
        return weights
        

    def update_critic(
            self, 
            obs_batch: torch.Tensor, 
            act_batch: torch.Tensor, 
            rew_batch: torch.Tensor, 
            next_obs_batch: torch.Tensor, 
            done_batch: torch.Tensor):

        with torch.no_grad():
            next_actions_dist = self.target_policy({'obs': next_obs_batch})
            next_actions = next_actions_dist.rsample(torch.Size([self.num_action_samples]))
            
            q_pred_next = self.target_critic(
                {'obs': next_obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
                next_actions).mean(0)
            
            qf_target = rew_batch +  (1. - done_batch.float()) * self.discount * q_pred_next

        qf_pred = self.critic({'obs': obs_batch}, act_batch)
        self.critic_optimizer.zero_grad()
        qf_loss = F.mse_loss(qf_pred,  qf_target, reduction='none').mean()
        qf_loss.backward()
        # clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optimizer.step()

        # avg_q_value = torch.min(qf1, qf2).mean()
        avg_q_value = qf_pred.mean().detach().item()
        avg_target_q_value = qf_target.mean().detach().item()
        
        return qf_loss.item(), avg_q_value, avg_target_q_value