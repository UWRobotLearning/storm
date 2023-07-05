import torch
from typing import Optional, Any

class ExperienceCollector(object):
    def __init__(self, cfg, policy, env, buffer):
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.episode_length = cfg['env']['episodeLength']


    # def collect_experience(self, 
    #                        num_steps_per_env: Optional[int]=None, 
    #                        update_buffer:bool = True, 
    #                        random_exploration:bool = False):
    def collect_experience(self, 
                           num_episodes: Optional[int]=None, 
                           update_buffer:bool = True, 
                           random_exploration:bool = False):

        total_steps_collected = 0
        total_episodes_done = 0
        # max_steps = num_steps_per_env * self.envs.num_envs
        
        if self.obs_dict is None:
            self.obs_dict = self.envs.reset()
        
        obs_dict = self.obs_dict
        
        if 'goal' in obs_dict:
            self.policy.update_goal(obs_dict['goal'])

        while total_episodes_done < num_episodes:
            with torch.no_grad():
                # if random_exploration:
                #     action = self.action_space.sample()
                # else:
                action = self.policy.get_action(obs_dict)
                next_obs_dict, reward, done, info = self.envs.step(action)
            
            self.curr_rewards += reward
            self.episode_lens += 1
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)
            done_episode_rewards = self.curr_rewards[done_indices]
            # self.done_episodes_reward_sum += torch.sum(done_episode_rewards).item()
            num_episodes_done = torch.sum(done).item()
            if num_episodes_done > 0:
                # print(self.curr_idx, num_episodes_done)
                # rem = min(10 - self.curr_idx, num_episodes_done)

                # if num_episodes_done > rem:
                #     #add to front
                #     extra = num_episodes_done - rem
                #     print(extra, rem)

                #     self.episode_reward_buffer[0:extra] = done_episode_rewards[-extra:]
                # self.episode_reward_buffer[self.curr_idx:self.curr_idx + rem] = done_episode_rewards[0:rem]
                # self.curr_idx = (self.curr_idx + num_episodes_done) % 10
                for i in range(num_episodes_done):
                    self.episode_reward_buffer.append(done_episode_rewards[i].item())
                    if len(self.episode_reward_buffer) > 10:
                        self.episode_reward_buffer.pop(0)


                self.total_episodes_done += num_episodes_done
            
            not_done = 1.0 - done.float()
            self.curr_rewards = self.curr_rewards * not_done
            self.episode_lens = self.episode_lens * not_done

            #remove timeout from done
            timeout = self.episode_lens == self.envs.max_episode_length
            done = done * (1-timeout.float())

            if update_buffer:
                #TODO: Save full actions
                self.buffer.add(obs_dict['obs'], self.envs.actions, reward, next_obs_dict['obs'], done)


            curr_num_steps = reward.shape[0]
            total_steps_collected += curr_num_steps
            self.obs_dict = obs_dict = copy.deepcopy(next_obs_dict)


            # if total_steps_collected >= max_steps:
            #     break
        
        if len(self.episode_reward_buffer) > 0:
            self.avg_episode_reward = np.average(self.episode_reward_buffer).item()
            # self.avg_episode_reward = self.done_episodes_reward_sum / self.total_episodes_done
        
        metrics = {
            'num_steps_collected': total_steps_collected,
            'buffer_size': len(self.buffer),
            # 'episode_reward_running_sum':self.done_episodes_reward_sum,
            'num_eps_completed': self.total_episodes_done,
            'avg_episode_reward': self.avg_episode_reward,
            # 'curr_steps_reward': self.curr_rewards.mean().item()
            }
        return metrics

    def update_policy_params(self, policy_param_dict):
        pass