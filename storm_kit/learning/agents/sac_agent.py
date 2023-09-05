import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
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
        self.num_action_samples = self.cfg['num_action_samples']
        self.num_train_episodes_per_epoch = self.cfg['num_train_episodes_per_epoch']
        self.num_updates_per_epoch = self.cfg['num_updates_per_epoch']
        self.automatic_entropy_tuning = self.cfg['automatic_entropy_tuning']

        if self.automatic_entropy_tuning:
            self.log_alpha = nn.Parameter(torch.tensor(self.cfg['init_log_alpha']))
            self.target_entropy = -np.prod(self.action_dim)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.cfg['alpha_optimizer']['lr'])
        else:
            self.alpha = torch.tensor(self.cfg['fixed_alpha'])
            self.log_alpha = torch.log(self.alpha)


    def train(self, model_dir=None):
        num_epochs = self.cfg['num_epochs']
        total_env_steps = 0

        # self.best_policy = copy.deepcopy(self.policy)
        # best_policy_perf = -torch.inf
        # best_policy_step = 0
        
        pbar = tqdm(range(int(num_epochs)), desc='train')
        for i in pbar:
            # step_start_time = time.time()
            #collect new experience
            self.buffer, play_metrics = self.runner_fn(
                envs=self.envs,
                num_episodes=self.num_train_episodes_per_epoch, 
                policy=self.policy,
                task=self.task,
                buffer=self.buffer,
                device=self.device
            )

            total_env_steps += play_metrics['num_steps_collected']
            #update agent
            for _ in range(self.num_updates_per_epoch):
                batch = self.buffer.sample(self.cfg['train_batch_size'])
                train_metrics = self.update(batch)
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

    def update(self, batch_dict):
        batch_dict = dict_to_device(batch_dict, self.device)

        #Update critic
        # self.critic_optimizer.zero_grad()
        critic_loss, avg_q_value = self.compute_critic_loss(batch_dict)
        if torch.any(torch.isnan(critic_loss)):
            import pdb; pdb.set_trace()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # #Update target critic using exponential moving average
        # for (param1, param2) in zip(self.target_critic.parameters(), self.critic.parameters()):
        #     param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)

        #Update policy
        # self.policy_optimizer.zero_grad()
        policy_loss, log_pi_new_actions = self.compute_policy_loss(batch_dict)
        if torch.any(torch.isnan(policy_loss)):
            import pdb; pdb.set_trace()
        # policy_loss.backward()
        # self.policy_optimizer.step()

        #Update temperature
        alpha_loss = torch.tensor([0.0])
        if self.automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -self.log_alpha * (log_pi_new_actions + self.target_entropy).detach()
            if torch.any(torch.isnan(policy_loss)):
                import pdb; pdb.set_trace()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        #Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        #Update Q function
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #Update target critic using exponential moving average
        for (param1, param2) in zip(self.target_critic.parameters(), self.critic.parameters()):
            param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)

            
        train_metrics = {}
        train_metrics['critic_loss'] = critic_loss.item()
        train_metrics['policy_loss'] = policy_loss.item()
        train_metrics['policy_entropy'] = log_pi_new_actions.item()
        train_metrics['avg_q_value'] = avg_q_value.item()
        train_metrics['alpha_loss'] = alpha_loss.item()
        train_metrics['alpha'] = torch.exp(self.log_alpha).item()
        return train_metrics


    # def compute_policy_loss(self, obs_batch):
    def compute_policy_loss(self, batch_dict):
        policy_input = {
            'states': batch_dict['state_dict'],
            'obs': batch_dict['obs'],
        }

        # new_action_dist = self.policy({'obs': obs_batch})
        # new_action_dist = self.policy(policy_input)
        # new_actions = new_action_dist.rsample(torch.Size([self.num_action_samples]))
        # log_pi_new_actions = new_action_dist.log_prob(new_actions).sum(-1).mean(0)
        new_actions = self.policy.get_action(policy_input, num_samples=self.num_action_samples)
        log_pi_new_actions = self.policy.log_prob(policy_input, new_actions)
        log_pi_new_actions = log_pi_new_actions.mean() #.sum(-1).mean(0)
        
        self.critic.requires_grad_(False)
        q_pred = self.critic(
            {'obs': batch_dict['obs'].unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
            new_actions).mean(0)

        # q_pred = self.critic(
        #     {'obs': obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
        #     new_actions).mean(0)
        
        alpha = self.log_alpha.exp() #torch.exp(self.log_alpha).item()
        self.critic.requires_grad_(True)
        policy_loss = (alpha * log_pi_new_actions - q_pred).mean() 
        if torch.any(torch.isnan(policy_loss)):
            print('in policy loss calc')
            import pdb; pdb.set_trace()



        return policy_loss, log_pi_new_actions.mean()


    def compute_critic_loss(self, batch_dict):
        cost_batch = batch_dict['cost'].squeeze(-1)
        obs_batch = batch_dict['obs']
        act_batch = batch_dict['action_dict']
        next_obs_batch = batch_dict['next_obs']
        next_state_batch = batch_dict['next_state_dict']
        done_batch = batch_dict['done'].squeeze(-1).float()
        rew_batch = -1.0 * cost_batch #TODO: fix

        with torch.no_grad():
            policy_input = {
                'states': next_state_batch,
                'obs': next_obs_batch}
            
            # next_actions_dist = self.policy(policy_input)
            next_actions = self.policy.get_action(policy_input, num_samples=self.num_action_samples)
            next_actions_log_prob = self.policy.log_prob(policy_input, next_actions).mean()
            # next_actions_log_prob = next_actions_log_prob.mean() #.sum(-1).mean(0)
            # next_actions = next_actions_dist.rsample(torch.Size([self.num_action_samples]))
            # next_actions_log_prob = next_actions_dist.log_prob(next_actions).sum(-1).mean(0)
            target_pred = self.target_critic(
                {'obs': next_obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
                next_actions).mean()
            
            alpha = self.log_alpha.exp() #torch.exp(self.log_alpha).item()
            q_pred_next =  target_pred - alpha * next_actions_log_prob
            q_target = rew_batch +  (1. - done_batch) * self.discount * q_pred_next

        qf1, qf2 = self.critic.both({'obs': obs_batch}, act_batch)
        qf1_loss = F.mse_loss(qf1,  q_target, reduction='none').mean()
        qf2_loss = F.mse_loss(qf2, q_target, reduction='none').mean()
        qf_loss = qf1_loss + qf2_loss

        avg_q_value = torch.min(qf1, qf2).mean()
        
        return qf_loss, avg_q_value


        