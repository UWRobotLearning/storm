import copy
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from storm_kit.learning.agents import Agent
import time
from tqdm import tqdm

class BCAgent(Agent):
    def __init__(
        self,
        cfg,
        envs,
        obs_space, 
        action_space,
        buffer,
        policy,
        critic=None,
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
        # optimizer_class = self.cfg['optimizer']
        self.optimizer = optim.Adam(self.policy.parameters(), 
                                    lr=self.cfg['optimizer']['lr'])
        self.loss_type = self.cfg['loss_type']
        self.num_action_samples = self.cfg['num_action_samples']
        self.fixed_alpha = self.cfg['fixed_alpha']
        self.best_policy = copy.deepcopy(self.policy)

    
    def train(self, model_dir=None):
        num_train_steps = self.cfg['num_train_steps']
        self.best_policy = copy.deepcopy(self.policy)
        best_policy_perf = -torch.inf
        best_policy_step = 0
        
        pbar = tqdm(range(int(num_train_steps)), desc='train')

        for i in pbar:
            # step_start_time = time.time()
            batch = self.buffer.sample(self.cfg['train_batch_size'])
            
            # update_start_time = time.time()
            train_metrics = self.update(*batch)
            # update_time = time.time() - update_start_time
            pbar.set_postfix(train_metrics)

            if (i % self.log_freq == 0) or (i == num_train_steps -1):
                if self.tb_writer is not None:
                    for k, v in train_metrics.items():
                        self.tb_writer.add_scalar('Train/' + k, v, i)
                        
            eval_metrics = {}
            if (i % self.eval_freq == 0) or (i == num_train_steps -1):
                # eval_start_time = time.time()
                eval_metrics = self.evaluate(num_eval_episodes= self.cfg['num_eval_episodes'])
                if self.logger is not None:
                    self.logger.row(eval_metrics, nostdout=True)
                if self.tb_writer is not None:
                    for k, v in eval_metrics.items():
                        self.tb_writer.add_scalar('Eval/' + k, v, i)

                if eval_metrics['eval_episode_reward_avg'] >= best_policy_perf:
                    self.best_policy = copy.deepcopy(self.policy)
                    best_policy_perf = eval_metrics['eval_episode_reward_avg']
                    best_policy_step = i

                self.policy.train()
                pbar.set_postfix(eval_metrics)
                # eval_time = time.time() - eval_start_time
                # print(f'Iter {i}, eval_time = {eval_time}')


            if (i % self.checkpoint_freq == 0) or (i == num_train_steps -1):
                print(f'Iter {i}: Saving best policy with average reward {best_policy_perf} from iteration {best_policy_step}')
                self.save(model_dir, save_buffer=False, best_policy=self.best_policy)
            
            # step_time = time.time() - step_start_time
            # print(f'Iter {i}, update_time = {update_time}, step_time = {step_time}')

    def update(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, state_batch=None, next_state_batch=None):
        obs_batch = obs_batch.to(self.device)
        act_batch = act_batch.to(self.device)
        
        self.optimizer.zero_grad()
        if self.loss_type == 'mse':
            new_actions = self.policy.get_action({'obs':obs_batch}, deterministic=False, num_samples=self.num_action_samples)            
            if new_actions.dim() == 3:
                act_batch = act_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)
            policy_imitation_loss = F.mse_loss(new_actions, act_batch, reduction='none')
            policy_imitation_loss = policy_imitation_loss.sum(-1).mean()
            policy_loss = policy_imitation_loss
        
        elif self.loss_type == 'nll':
            action_dist = self.policy({'obs':obs_batch})
            policy_imitation_loss = -1.0 * action_dist.log_prob(act_batch).mean()
            #compute policy entropy
            new_actions = action_dist.rsample()
            log_prob_new_actions = action_dist.log_prob(new_actions)
            policy_entropy = -log_prob_new_actions.mean()
            policy_loss = policy_imitation_loss - self.fixed_alpha * policy_entropy
        else:
            raise NotImplementedError('Unidentified loss type.')
        
        policy_loss.backward()
        self.optimizer.step()

        metrics = {}
        metrics['policy_imitation_loss'] = policy_imitation_loss.item()
        metrics['policy_loss'] = policy_loss.item()
        metrics['policy_entropy'] = policy_entropy.item()
        
        return metrics



    def compute_policy_loss(self):
        pass


    def save(self, model_dir, save_buffer=False, iter=0, best_policy=None):
        state = {
            'iter': iter,
            'policy_state_dict': self.policy.state_dict(),
            'best_policy_state_dict': best_policy.state_dict()
        }
        torch.save(state, os.path.join(model_dir, 'agent_checkpoint_{}.pt'.format(iter)))
        if save_buffer:
            print('Saving buffer len= {}'.format(len(self.buffer)))
            self.buffer.save(os.path.join(model_dir, 'agent_buffer_{}.pt'.format(iter)))
    
    def load(self, checkpoint_path, buffer_path=None):
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.best_policy.load_state_dict(checkpoint['best_policy_state_dict'])
        if buffer_path is not None:
            self.buffer.load(buffer_path)

