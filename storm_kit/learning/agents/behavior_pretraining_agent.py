import copy
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.profiler import record_function

from storm_kit.learning.agents import Agent
from storm_kit.learning.learning_utils import dict_to_device
import time
from tqdm import tqdm

class BPAgent(Agent):
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
            cfg, envs, task, obs_dim, action_dim, #obs_space, action_space,
            buffer=buffer, policy=policy,
            runner_fn=runner_fn,
            logger=logger, tb_writer=tb_writer,
            device=device
        )
        self.critic = critic
        # optimizer_class = self.cfg['optimizer']
        self.policy_optimizer = optim.Adam(self.policy.parameters(), 
                                    lr=self.cfg['policy_optimizer']['lr'])
        self.policy_loss_type = self.cfg['policy_loss_type']
        self.num_action_samples = self.cfg['num_action_samples']
        self.fixed_alpha = self.cfg['fixed_alpha']
        if self.policy_loss_type not in ["mse", "nll"]:
            raise ValueError('Unidentified policy loss type {}.'.format(self.policy_loss_type))
        self.num_eval_episodes = self.cfg.get('num_eval_episodes', 1)
        self.eval_first_policy = self.cfg.get('eval_first_policy', False)
        self.policy_use_tanh = self.cfg.get('policy_use_tanh', False)
        # self.best_policy = copy.deepcopy(self.policy)

    
    def train(self, model_dir=None, data_dir=None):
        num_train_steps = self.cfg['num_pretrain_steps']
        # self.best_policy = copy.deepcopy(self.policy)
        # best_policy_perf = -torch.inf
        # best_policy_step = 0
        
        pbar = tqdm(range(int(num_train_steps)), desc='train')

        for i in pbar:
            #Evaluate policy at some frequency
            if ((i + (1-self.eval_first_policy)) % self.eval_freq == 0) or (i == num_train_steps -1):
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

                # if eval_metrics['eval_episode_reward_avg'] >= best_policy_perf:
                #     self.best_policy = copy.deepcopy(self.policy)
                #     best_policy_perf = eval_metrics['eval_episode_reward_avg']
                #     best_policy_step = i

                self.policy.train()
                pbar.set_postfix(eval_metrics)

            with record_function('sample_batch'):
                batch = self.buffer.sample(self.cfg['train_batch_size'])
                batch = dict_to_device(batch, self.device)
            
            if self.relabel_data:
                with record_function('relabel_data'):
                    batch = self.relabel_batch(batch)
            
            with record_function('update'):
                train_metrics = self.update(batch, i)
            
            pbar.set_postfix(train_metrics)

            if (i % self.log_freq == 0) or (i == num_train_steps -1):
                if self.tb_writer is not None:
                    for k, v in train_metrics.items():
                        self.tb_writer.add_scalar('Train/' + k, v, i)
                        
            if (i % self.checkpoint_freq == 0) or (i == num_train_steps -1):
                print(f'Iter {i}: Saving current policy')
                self.save(model_dir, data_dir, iter=i)
            
            # step_time = time.time() - step_start_time
            # print(f'Iter {i}, update_time = {update_time}, step_time = {step_time}')

    def update(self, batch_dict, step_num):

        train_metrics = {}

        self.policy_optimizer.zero_grad()
        policy_loss, log_pi_new_actions = self.compute_policy_loss(batch_dict)
        policy_loss.backward()
        self.policy_optimizer.step()
        train_metrics['policy_loss'] = policy_loss.item()
        train_metrics['policy_entropy'] = log_pi_new_actions.item()
        
        return train_metrics



    def compute_policy_loss(self, batch_dict):
        
        obs_batch = batch_dict['obs']
        state_batch =batch_dict['state_dict']
        act_batch = batch_dict['actions']
        if self.policy_use_tanh:
            act_batch = torch.tanh(act_batch)

        if self.policy_loss_type == 'mse':
            new_actions = self.policy.get_action({'obs':obs_batch}, deterministic=False, num_samples=self.num_action_samples)            
            if new_actions.dim() == 3:
                act_batch = act_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)
            policy_imitation_loss = F.mse_loss(new_actions, act_batch, reduction='none')
            policy_imitation_loss = policy_imitation_loss.sum(-1).mean()
            policy_loss = policy_imitation_loss
        
        elif self.policy_loss_type == 'nll':
            policy_input = {
                'states': state_batch,
                'obs': obs_batch}
            policy_loss = -1.0 * self.policy.log_prob(policy_input, act_batch).mean()
            # action_dist = self.policy({'obs':obs_batch})
            # policy_imitation_loss = -1.0 * action_dist.log_prob(act_batch).mean()
            # #compute policy entropy
            # new_actions = action_dist.rsample()
            # log_prob_new_actions = action_dist.log_prob(new_actions)
            # policy_entropy = -log_prob_new_actions.mean()
            # policy_loss = policy_imitation_loss - self.fixed_alpha * policy_entropy

        new_actions, log_pi_new_actions = self.policy.entropy(policy_input)


        return policy_loss, log_pi_new_actions.mean()


    # def save(self, model_dir, save_buffer=False, iter=0, best_policy=None):
    #     state = {
    #         'iter': iter,
    #         'policy_state_dict': self.policy.state_dict(),
    #         'best_policy_state_dict': best_policy.state_dict()
    #     }
    #     torch.save(state, os.path.join(model_dir, 'agent_checkpoint_{}.pt'.format(iter)))
    #     if save_buffer:
    #         print('Saving buffer len= {}'.format(len(self.buffer)))
    #         self.buffer.save(os.path.join(model_dir, 'agent_buffer_{}.pt'.format(iter)))
    
    # def load(self, checkpoint_path, buffer_path=None):
    #     checkpoint = torch.load(checkpoint_path)
    #     self.policy.load_state_dict(checkpoint['policy_state_dict'])
    #     self.best_policy.load_state_dict(checkpoint['best_policy_state_dict'])
    #     if buffer_path is not None:
    #         self.buffer.load(buffer_path)

