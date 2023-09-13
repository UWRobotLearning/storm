import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from storm_kit.learning.agents import Agent, SACAgent
from storm_kit.learning.learning_utils import dict_to_device


class MPQAgent(SACAgent):
    def __init__(
            self,
            cfg,
            envs,
            task,
            obs_dim, 
            action_dim,
            buffer,
            policy,
            mpc_policy,
            target_mpc_policy,
            critic,
            runner_fn,
            state_filter=None,
            logger=None,
            tb_writer=None,
            device=torch.device('cpu'),
    ):


        super().__init__(
            cfg, envs, task, obs_dim, action_dim,
            buffer=buffer, policy=policy, critic=critic,
            runner_fn=runner_fn, state_filter=state_filter, logger=logger, 
            tb_writer=tb_writer, device=device        
        )
        self.mpc_policy = mpc_policy
        self.target_mpc_policy = target_mpc_policy
        self.collect_data_with_mpc = cfg.collect_data_with_mpc


    def collect_experience(self):
        if self.collect_data_with_mpc:
            self.buffer, play_metrics = self.runner_fn(
                envs=self.envs,
                num_episodes=self.num_train_episodes_per_epoch, 
                policy=self.mpc_policy,
                task=self.task,
                state_filter = self.state_filter,
                buffer=self.buffer,
                device=self.device
            )
            return play_metrics
        else:
            return super().collect_experience()


    # def update(self, batch_dict):
    #     batch_dict = dict_to_device(batch_dict, self.device)

    #     #Update critic
    #     self.critic_optimizer.zero_grad()
    #     critic_loss, avg_q_value = self.compute_critic_loss(batch_dict)
    #     critic_loss.backward()
    #     self.critic_optimizer.step()
    #     #Update target critic using exponential moving average
    #     for (param1, param2) in zip(self.target_critic.parameters(), self.critic.parameters()):
    #         param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)

    #     # #Update policy
    #     # self.policy_optimizer.zero_grad()
    #     # policy_loss, log_pi_new_actions = self.compute_policy_loss(batch_dict)
    #     # policy_loss.backward()
    #     # self.policy_optimizer.step()

    #     # #Update temperature
    #     # alpha_loss = torch.tensor([0.0])
    #     # if self.automatic_entropy_tuning:
    #     #     self.alpha_optimizer.zero_grad()
    #     #     alpha_loss = self.log_alpha * (-log_pi_new_actions.detach() - self.target_entropy)
    #     #     alpha_loss.backward()
    #     #     self.alpha_optimizer.step()

            
    #     train_metrics = {}
    #     train_metrics['critic_loss'] = critic_loss.item()
    #     # train_metrics['policy_loss'] = policy_loss.item()
    #     # train_metrics['policy_entropy'] = log_pi_new_actions.item()
    #     train_metrics['avg_q_value'] = avg_q_value.item()
    #     # train_metrics['alpha_loss'] = alpha_loss.item()
    #     train_metrics['alpha'] = torch.exp(self.log_alpha).item()
    #     return train_metrics




    # def compute_critic_loss(self, batch_dict):
    #     cost_batch = batch_dict['cost'].squeeze(-1)
    #     obs_batch = batch_dict['obs']
    #     act_batch = batch_dict['action_dict']
    #     next_obs_batch = batch_dict['next_obs']
    #     next_state_batch = batch_dict['next_state_dict']
    #     done_batch = batch_dict['done'].squeeze(-1)
    #     # def compute_bellman_target(q_pred_next):
    #     #     assert cost_batch.shape == q_pred_next.shape
    #     #     return (cost_batch + (1.-done_batch.float())*self.discount*q_pred_next)#.clamp(min=self._Vmin, max=self._Vmax)
    #     # rew_batch = -1.0 * cost_batch #TODO: fix
    #     with torch.no_grad():
    #         policy_input = {
    #             'states': next_state_batch,
    #             'obs': next_obs_batch}
            
    #         # next_actions_dist = self.policy(policy_input)
    #         self.target_policy.reset()
    #         next_actions, next_actions_log_prob = self.target_policy.entropy(policy_input)
    #         # next_actions_log_prob = self.policy.log_prob(policy_input, next_actions)
    #         # next_actions_log_prob = next_actions_log_prob.sum(-1).mean(0)
    #         # next_actions = next_actions_dist.rsample(torch.Size([self.num_action_samples]))
    #         # next_actions_log_prob = next_actions_dist.log_prob(next_actions).sum(-1).mean(0)
    #         target_pred = self.target_critic(
    #             {'obs': next_obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
    #             next_actions).mean(0)
            
    #         # alpha = torch.exp(self.log_alpha).item()
    #         q_pred_next =  target_pred #- alpha * next_actions_log_prob
    #         q_target = cost_batch +  (1. - done_batch.float()) * self.discount * q_pred_next

    #     qf1, qf2 = self.critic.both({'obs': obs_batch}, act_batch)
    #     qf1_loss = F.mse_loss(qf1,  q_target, reduction='none').mean()
    #     qf2_loss = F.mse_loss(qf2, q_target, reduction='none').mean()
    #     qf_loss = qf1_loss + qf2_loss

    #     avg_q_value = torch.min(qf1, qf2).mean()
        
    #     return qf_loss, avg_q_value