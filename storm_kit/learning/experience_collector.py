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
    def collect_episodes(self, 
                        num_episodes_per_env: Optional[int]=None,  
                        policy:Optional[Any] = None,
                        data_folder: str = None):

        collect_data = False
        if data_folder is not None:
            collect_data = True
        
        # total_steps_collected = 0

        # max_steps = num_steps_per_env * self.envs.num_envs
        max_
        
        if self.state_dict is None:
            self.state_dict = self.envs.reset()
            obs_dict = {
                'states': self.state_dict
            }
        if self.targets is None:
            self.targets = self.task.get_randomized_goals(
                num_envs = self.envs.num_envs, randomization_mode='randomize_position')
        
        policy.update_goal(self.targets)

        while True:
            with torch.no_grad():
                # if random_exploration:
                #     action = self.action_space.sample()
                # else:
                obs_dict = {
                    'states': self.state_dict
                }

                action = policy.get_action(obs_dict)
                # next_obs_dict, reward, done, info = self.envs.step(action)
                next_state_dict, done = self.envs.step(action)
            
            self.curr_rewards += 1.0
            self.episode_lens += 1
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)
            done_episode_rewards = self.curr_rewards[done_indices]
            #reset goal poses
            if len(done_indices) > 0:
                self.targets = self.task.get_randomized_goals(
                    num_envs = self.envs.num_envs, 
                    target_pose_buff=self.targets,
                    env_ids=done_indices,
                    randomization_mode='randomize_position')
                policy.update_goal(self.targets)


            # self.done_episodes_reward_sum += torch.sum(done_episode_rewards).item()
            num_episodes_done = torch.sum(done).item()
            if num_episodes_done > 0:
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
            timeout = self.episode_lens == self.envs.max_episode_length - 1
            done = done * (1-timeout.float())

            # if update_buffer:
            #     #TODO: Save full actions
            #     self.buffer.add(obs_dict['obs'], self.envs.actions, reward, next_obs_dict['obs'], done)

            # curr_num_steps = reward.shape[0]
            curr_num_steps = self.state_dict['q_pos'].shape[0]
            total_steps_collected += curr_num_steps

            # self.obs_dict = obs_dict = copy.deepcopy(next_obs_dict)
            self.state_dict = copy.deepcopy(next_state_dict)

            if total_steps_collected >= max_steps:
                break
        
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
    
print('Collecting episodes')        
input('Press any key to initiate reset')
self.reset()
for ep_num in range(num_episodes):
    input('Press any key to start collecting episode = {}'.format(ep_num))
    episode_buffer = self.collect_episode(collect_data=collect_data)
    if data_folder is not None:
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        filepath = data_folder + '/episode_{}.p'.format(ep_num)
        print('Saving episode to {}'.format(filepath))
        episode_buffer.save(filepath)

    input('Episode {}/{} done. Press any key to initiate reset'.format(ep_num, num_episodes))
    self.reset()



    def update_policy_params(self, policy_param_dict):
        pass