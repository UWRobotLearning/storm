from typing import Optional
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.profiler import record_function
import torch.nn as nn
from tqdm import tqdm
from storm_kit.learning.agents import Agent
from storm_kit.learning.learning_utils import dict_to_device


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
            init_buffer=None,
            target_critic=None,
            logger=None,
            tb_writer=None,
            device=torch.device('cpu'),
            train_rng:Optional[torch.Generator]=None,
            eval_rng:Optional[torch.Generator]=None
    ):


        super().__init__(
            cfg, envs, task, obs_dim, action_dim,
            buffer=buffer, policy=policy,
            runner_fn=runner_fn,
            logger=logger, tb_writer=tb_writer, 
            device=device, train_rng=train_rng,
            eval_rng=eval_rng        
        )
        self.init_buffer = init_buffer #initial dataset
        self.critic = critic
        # self.target_critic = copy.deepcopy(self.critic)
        self.target_critic = target_critic
        self.policy_optimizer = optim.Adam(self.policy.parameters(), 
                                    lr=float(self.cfg['policy_optimizer']['lr']))
        self.critic_optimizer =  optim.Adam(self.critic.parameters(), 
                                    lr=float(self.cfg['critic_optimizer']['lr']))
        self.polyak_tau = float(self.cfg['polyak_tau'])
        self.discount = self.cfg['discount']
        self.num_action_samples = self.cfg['num_action_samples']
        self.num_train_episodes_per_epoch = self.cfg['num_train_episodes_per_epoch']
        self.num_updates_per_epoch = self.cfg.get('num_updates_per_epoch', None)
        self.update_to_data_ratio = self.cfg['update_to_data_ratio']
        self.policy_update_delay = self.cfg['policy_update_delay']
        self.automatic_entropy_tuning = self.cfg['automatic_entropy_tuning']
        self.backup_entropy = self.cfg['backup_entropy']
        self.min_buffer_size = int(self.cfg['min_buffer_size'])
        self.reward_scale = self.cfg['reward_scale']
        self.num_eval_episodes = self.cfg.get('num_eval_episodes', 1)
        self.eval_first_policy = self.cfg.get('eval_first_policy', False)


        if self.automatic_entropy_tuning:
            self.log_alpha = nn.Parameter(torch.tensor(self.cfg['init_log_alpha']))
            self.target_entropy = -np.prod(self.action_dim)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(self.cfg['alpha_optimizer']['lr']))
        else:
            self.alpha = torch.tensor(self.cfg['fixed_alpha'])
            self.log_alpha = torch.log(self.alpha)


    def train(self, debug:bool=False, model_dir=None):
        num_epochs = int(self.cfg['num_epochs'])
        total_env_steps = 0

        # self.best_policy = copy.deepcopy(self.policy)
        # best_policy_perf = -torch.inf
        # best_policy_step = 0
        
        pbar = tqdm(range(int(num_epochs)), desc='train')
        for i in pbar:

            #Evaluate policy at some frequency
            if ((i + (1-self.eval_first_policy)) % self.eval_freq == 0) or (i == num_epochs -1):
                print('Evaluating policy')
                self.policy.eval()
                eval_metrics = self.evaluate_policy(
                    self.policy, 
                    num_eval_episodes=self.num_eval_episodes, 
                    deterministic=True, 
                    debug=False)
                
                print(eval_metrics)
                if self.logger is not None:
                    self.logger.row(eval_metrics, nostdout=True)
                if self.tb_writer is not None:
                    for k, v in eval_metrics.items():
                        self.tb_writer.add_scalar('Eval/' + k, v, i)

            #collect new experience
            new_buffer, play_metrics = self.collect_experience(
                policy=self.policy,
                num_episodes=self.num_train_episodes_per_epoch, 
                update_buffer=True, 
                deterministic=False,
                debug=debug)
                    
            self.buffer.concatenate(new_buffer.qlearning_dataset())
            print(play_metrics, 'Buffer len = {}'.format(len(self.buffer)))
            num_steps_collected = play_metrics['num_steps_collected'] 
            total_env_steps += num_steps_collected

            #update
            self.policy.train()
            self.critic.train()
            if len(self.buffer) >= self.min_buffer_size:                
                if self.num_updates_per_epoch is not None:
                    #use fixed number of updates
                    num_update_steps = int(self.num_updates_per_epoch)
                elif self.update_to_data_ratio is not None:
                    #use number of updates based on amount of data collected
                    num_update_steps = int(self.update_to_data_ratio * num_steps_collected)
                else:
                    raise ValueError('Either num_updates_per_epoch or update_to_data_ratio must be provided')
                
                print('Running {} updates'.format(num_update_steps))

                for k in range(num_update_steps):
                    with record_function('sample_batch'):
                        batch = self.buffer.sample(self.cfg['train_batch_size'])
                        batch = dict_to_device(batch, self.device)
                    
                    if self.relabel_data:
                        with record_function('relabel_data'):
                            batch = self.preprocess_batch(batch, compute_cost_and_terminals=True)

                    # batch = self.buffer.sample(self.cfg['train_batch_size'])
                    with record_function('update'):
                        train_metrics = self.update(batch, k, num_update_steps)
                    pbar.set_postfix(train_metrics)

                if (i % self.log_freq == 0) or (i == num_epochs -1):
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

    # def collect_experience(self, debug:bool = False):

    #     self.buffer, play_metrics = self.runner_fn(
    #         envs=self.envs,
    #         num_episodes=self.num_train_episodes_per_epoch, 
    #         policy=self.policy,
    #         task=self.task,
    #         buffer=self.buffer,
    #         debug = debug,
    #         device=self.device
    #     )
    #     return play_metrics

    def update(self, batch_dict, step_num, num_update_steps):
        batch_dict = dict_to_device(batch_dict, self.device)
        train_metrics = {}

        #Compute critic loss
        self.critic_optimizer.zero_grad()
        critic_loss, avg_q_value = self.compute_critic_loss(batch_dict)
        critic_loss.backward()
        self.critic_optimizer.step()


        if ((step_num + 1) % self.policy_update_delay == 0) or (step_num == num_update_steps - 1): 
            #Compute policy loss
            self.policy_optimizer.zero_grad()
            policy_loss, log_pi_new_actions = self.compute_policy_loss(batch_dict)
            policy_loss.backward()
            self.policy_optimizer.step()
            train_metrics['policy_loss'] = policy_loss.item()
            train_metrics['policy_entropy'] = log_pi_new_actions.item()

            #Update temperature
            alpha_loss = torch.tensor([0.0])
            if self.automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss = -self.log_alpha * (log_pi_new_actions + self.target_entropy).detach()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                train_metrics['alpha_loss'] = alpha_loss.item()

        #Update target critic using exponential moving average
        for (param1, param2) in zip(self.target_critic.parameters(), self.critic.parameters()):
            param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)

        train_metrics['critic_loss'] = critic_loss.item()
        train_metrics['avg_q_value'] = avg_q_value.item()
        train_metrics['alpha'] = torch.exp(self.log_alpha).item()

        return train_metrics


    def compute_policy_loss(self, batch_dict):
        policy_input = {
            'states': batch_dict['state_dict'],
            'obs': batch_dict['obs'],
        }

        new_actions, log_pi_new_actions = self.policy.entropy(policy_input, self.num_action_samples)
        log_pi_new_actions = log_pi_new_actions.mean(-1) #mean along action dimension

        self.critic.requires_grad_(False)
        q_pred = self.critic(
            {'obs': batch_dict['obs'].unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
            new_actions).mean(0) #mean along num_action_samples
        
        alpha = self.log_alpha.exp() #torch.exp(self.log_alpha).item()
        self.critic.requires_grad_(True)
        # policy_loss = (alpha * log_pi_new_actions - q_pred).mean() 
        policy_loss = (alpha * log_pi_new_actions + q_pred).mean() #signs flipped on Q since we are minimizing costs


        return policy_loss, log_pi_new_actions.mean()


    def compute_critic_loss(self, batch_dict):
        cost_batch = batch_dict['cost'].squeeze(-1)
        obs_batch = batch_dict['obs']
        act_batch = batch_dict['actions']
        next_obs_batch = batch_dict['next_obs']
        next_state_batch = batch_dict['next_state_dict']
        done_batch = batch_dict['terminals'].squeeze(-1).float()

        with torch.no_grad():
            policy_input = {
                'states': next_state_batch,
                'obs': next_obs_batch}
            
            next_actions, next_actions_log_prob = self.policy.entropy(policy_input)
            next_actions_log_prob = next_actions_log_prob.mean(-1) #mean along action dimension

            target_pred = self.target_critic(
                {'obs': next_obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
                next_actions) #.mean()
            target_pred = target_pred.mean(0) #mean across num action samples
            
            if self.backup_entropy:
                alpha = self.log_alpha.exp()
                q_pred_next =  target_pred + alpha * next_actions_log_prob # sign is flipped in entropy since we are minimizing costs
            else:
                q_pred_next = target_pred

            q_target = self.reward_scale * cost_batch +  (1. - done_batch) * self.discount * q_pred_next

        qf_all = self.critic.all({'obs': obs_batch}, act_batch)
        q_target = q_target.unsqueeze(-1).repeat(1, qf_all.shape[-1])

        qf_loss = F.mse_loss(qf_all, q_target, reduction='none')
        qf_loss = qf_loss.sum(-1).mean(0) #sum along ensemble dimension and mean along batch

        # qf_loss = 0.0
        # for q in qf_all:
        #     qf_loss += F.mse_loss(q, q_target, reduction='none').mean()
        # qf_loss = [F.mse_loss(q, q_target, reduction='none').mean() for q in qf_all]
        # qf_loss = torch.sum(*qf_loss)
        # qf1, qf2 = self.critic.both({'obs': obs_batch}, act_batch)
        # qf1_loss = F.mse_loss(qf1,  q_target, reduction='none').mean()
        # qf2_loss = F.mse_loss(qf2, q_target, reduction='none').mean()
        # qf_loss = qf1_loss + qf2_loss

        # avg_q_value = torch.min(qf1, qf2).mean()
        # avg_q_value = torch.min(qf_all, dim=-1)[0].mean()
        avg_q_value = torch.max(qf_all, dim=-1)[0].mean() #max instead of min since we are minimizing costs



        return qf_loss, avg_q_value


        