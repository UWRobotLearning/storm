import copy
import os
import pickle
from typing import Optional, Any
import torch
import torch.nn as nn
import numpy as np
from storm_kit.learning.learning_utils import Log


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
        # eval_envs=None,
        logger=None,
        tb_writer=None,
        device=torch.device('cpu'),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.envs = envs
        self.task = task
        # self.eval_envs = eval_envs
        self.policy = policy
        self.runner_fn = runner_fn
        self.buffer = buffer
        self.device = device
        self.logger = logger
        self.tb_writer = tb_writer
        self.obs_dict = None
        self.state_dict = None
        self.targets = None
        self.log_freq = self.cfg['log_freq']
        self.eval_freq = self.cfg['eval_freq']  
        self.checkpoint_freq = self.cfg['checkpoint_freq']
        self.init_buffers()


    def init_buffers(self):
        self.curr_rewards = torch.zeros(self.envs.num_envs, device=self.device)
        self.episode_lens = torch.zeros(self.envs.num_envs, device=self.device)
        # self.done_episodes_reward_sum = 0.0 
        self.episode_reward_buffer = []
        self.curr_idx = 0
        self.total_episodes_done = 0
        self.avg_episode_reward = 0.0


    def update(self):
        return {}

    def train(self, model_dir=None):
        # self.state_dict = self.envs.reset()
        for i in range (self.total_steps):
            episode_metrics = self.collect_experience(num_steps=self.cfg['steps_per_update'], update_buffer=True)
            train_metrics = self.update()

    def save(self, model_dir, save_buffer=False, iter=0):
        state = {
            'iter': iter,
            'policy_state_dict': self.policy.state_dict(),
        }
        torch.save(state, os.path.join(model_dir, 'agent_checkpoint_{}.pt'.format(iter)))
        if save_buffer:
            print('Saving buffer len= {}'.format(len(self.buffer)))
            self.buffer.save(os.path.join(model_dir, 'agent_buffer_{}.pt'.format(iter)))

    def load(self, checkpoint_path, buffer_path=None):
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if buffer_path is not None:
            self.buffer.load(buffer_path)
    
    # def collect_experience(self, 
    #                        num_steps_per_env: Optional[int]=None, 
    #                        buffer = None, 
    #                        policy:Optional[Any] = None,
    #                        random_exploration:bool = False):
        
    #     if buffer is not None:
    #         update_buffer = True

    #     if policy is None:
    #         policy = self.policy
        
    #     total_steps_collected = 0
    #     max_steps = num_steps_per_env * self.envs.num_envs
        
    #     if self.state_dict is None:
    #         self.state_dict = self.envs.reset()
    #         obs_dict = {
    #             'states': self.state_dict
    #         }
    #     if self.targets is None:
    #         self.targets = self.task.get_randomized_goals(
    #             num_envs = self.envs.num_envs, randomization_mode='randomize_position')
        
    #     policy.update_goal(self.targets)
    #     self.task.update_params(goal = self.targets)

    #     while True:
    #         with torch.no_grad():
    #             # if random_exploration:
    #             #     action = self.action_space.sample()
    #             # else:
    #             obs_dict = {
    #                 'states': self.state_dict
    #             }

    #             action = policy.get_action(obs_dict)
    #             # next_obs_dict, reward, done, info = self.envs.step(action)
    #             next_state_dict, done = self.envs.step(action)
    #             reward = 1.0
    #             obs = self.task.compute_observations(self.state_dict)
            
    #         self.curr_rewards += reward
    #         self.episode_lens += 1
    #         done_indices = done.nonzero(as_tuple=False).squeeze(-1)
    #         done_episode_rewards = self.curr_rewards[done_indices]
    #         #reset goal poses
    #         if len(done_indices) > 0:
    #             self.targets = self.task.get_randomized_goals(
    #                 num_envs = self.envs.num_envs, 
    #                 target_pose_buff=self.targets,
    #                 env_ids=done_indices,
    #                 randomization_mode='randomize_position')
    #             policy.update_goal(self.targets)
    #             self.task.update_params(self.targets)


    #         # self.done_episodes_reward_sum += torch.sum(done_episode_rewards).item()
    #         num_episodes_done = torch.sum(done).item()
    #         if num_episodes_done > 0:
    #             # rem = min(10 - self.curr_idx, num_episodes_done)

    #             # if num_episodes_done > rem:
    #             #     #add to front
    #             #     extra = num_episodes_done - rem
    #             #     print(extra, rem)

    #             #     self.episode_reward_buffer[0:extra] = done_episode_rewards[-extra:]
    #             # self.episode_reward_buffer[self.curr_idx:self.curr_idx + rem] = done_episode_rewards[0:rem]
    #             # self.curr_idx = (self.curr_idx + num_episodes_done) % 10
    #             for i in range(num_episodes_done):
    #                 self.episode_reward_buffer.append(done_episode_rewards[i].item())
    #                 if len(self.episode_reward_buffer) > 10:
    #                     self.episode_reward_buffer.pop(0)

    #             self.total_episodes_done += num_episodes_done
            
    #         not_done = 1.0 - done.float()
    #         self.curr_rewards = self.curr_rewards * not_done
    #         self.episode_lens = self.episode_lens * not_done

    #         #remove timeout from done
    #         timeout = self.episode_lens == self.envs.max_episode_length - 1
    #         done = done * (1-timeout.float())

    #         if update_buffer:
    #         #     #TODO: Save full actions
    #             buffer.add(
    #                 q_pos = self.state_dict['q_pos'],
    #                 q_vel = self.state_dict['q_vel'],
    #                 q_acc = self.state_dict['q_acc'],
    #                 q_pos_cmd = action['q_pos'],
    #                 q_vel_cmd = action['q_vel'],
    #                 q_acc_cmd = action['q_acc'],
    #                 ee_goal = self.targets
    #                 obs = obs, 
    #                 reward=reward, 
    #                 )

    #         # curr_num_steps = reward.shape[0]
    #         curr_num_steps = self.state_dict['q_pos'].shape[0]
    #         total_steps_collected += curr_num_steps

    #         # self.obs_dict = obs_dict = copy.deepcopy(next_obs_dict)
    #         self.state_dict = copy.deepcopy(next_state_dict)

    #         if total_steps_collected >= max_steps:
    #             break
        
    #     if len(self.episode_reward_buffer) > 0:
    #         self.avg_episode_reward = np.average(self.episode_reward_buffer).item()
    #         # self.avg_episode_reward = self.done_episodes_reward_sum / self.total_episodes_done
        
    #     metrics = {
    #         'num_steps_collected': total_steps_collected,
    #         'buffer_size': len(self.buffer),
    #         # 'episode_reward_running_sum':self.done_episodes_reward_sum,
    #         'num_eps_completed': self.total_episodes_done,
    #         'avg_episode_reward': self.avg_episode_reward,
    #         # 'curr_steps_reward': self.curr_rewards.mean().item()
    #         }
    #     return metrics
    
    # def evaluate(self, num_eval_episodes:int):
    #     #TODO: eval envs need to be seperate (ideally equal in number to num_eval_episodes)

    #     self.policy.eval()
    #     obs_dict =  self.envs.reset()
    #     total_episodes_done = 0
    #     if 'goal' in obs_dict:
    #         self.policy.update_goal(obs_dict['goal'])
        
    #     curr_rewards = torch.zeros(self.envs.num_envs, device=self.device)
    #     eval_episode_reward_sum = 0.0

    #     while total_episodes_done < num_eval_episodes:
    #         with torch.no_grad():
    #             action = self.policy.get_action(obs_dict, deterministic=True)
    #             next_obs_dict, reward, done, info = self.envs.step(action)
            
    #         curr_rewards += reward
    #         done_indices = done.nonzero(as_tuple=False)
    #         done_episode_rewards = curr_rewards[done_indices]
    #         eval_episode_reward_sum += torch.sum(done_episode_rewards).item()
    #         total_episodes_done += torch.sum(done).item()

    #         not_done = 1.0 - done.float()
    #         curr_rewards = self.curr_rewards * not_done

    #         obs_dict = copy.deepcopy(next_obs_dict)

    #     eval_episode_reward_avg = eval_episode_reward_sum / total_episodes_done*1.0


    #     metrics = {
    #         'eval_num_episodes': total_episodes_done,
    #         'eval_episode_reward_sum': eval_episode_reward_sum,
    #         'eval_episode_reward_avg': eval_episode_reward_avg,
    #         }

    #     return metrics
    
    # def evaluate(self, num_eval_episodes):
    #     self.policy.eval()
    #     obs_dict = self.eval_envs.reset()
    #     if 'goal' in obs_dict:
    #         self.policy.update_goal(obs_dict['goal'])  

    #     num_episodes_done = 0
    #     curr_rewards = torch.zeros(self.eval_envs.num_envs, device=self.device)
    #     episode_rewards = []

    #     while num_episodes_done < num_eval_episodes:
    #         with torch.no_grad():
    #             action = self.policy.get_action(obs_dict['obs'])
    #             next_obs_dict, reward, done, info = self.eval_envs.step(action)
    #         curr_rewards += reward

    #         done_indices = done.nonzero(as_tuple=False)
    #         done_ep_rews = curr_rewards[done_indices]
    #         episode_rewards.append()




    #         obs_dict = next_obs_dict    




    # def eval(self, num_episodes: int = 1):

    #     total_steps_collected = 0
    #     obs_dict = self.envs.reset()
    #     if 'goal' in obs_dict:
    #         self.policy.update_goal(obs_dict['goal'])  
        
    #     while True:
    #         action = self.policy.get_action(obs_dict)
    #         next_obs_dict, reward, done, info = self.envs.step(action)
    #         # self.buffer.add(obs_dict['obs'], action, reward, next_obs_dict['obs'], done)
    #         curr_num_steps = reward.shape[0]
    #         #reset the done environments and update goal
    #         total_steps_collected += curr_num_steps
    #         obs_dict = next_obs_dict

    #         if total_steps_collected >= num_steps:
    #             break
        

        # torch.save(self.policy.state_dict(), os.path.join(model_dir, 'policy_{}.pt'.format(iter)))

        # if load_buffer:
        #     torch.save(self.buffer, os.path.join(model_dir, 'buffer_{}.pt'.format(iter)))
