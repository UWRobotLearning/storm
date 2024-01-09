import copy
from collections import defaultdict
from typing import Optional
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.profiler import record_function
import torch.nn as nn
from tqdm import tqdm
from storm_kit.learning.agents import Agent
from storm_kit.learning.learning_utils import dict_to_device, plot_episode


class MPQAgent(Agent):
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
            target_critic,
            init_buffer=None,
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
        self.init_buffer = init_buffer
        if self.init_buffer is not None:
            self.buffer.concatenate(self.init_buffer.qlearning_dataset())
        self.critic = critic
        # self.old_policy = copy.deepcopy(self.policy)
        self.target_critic = target_critic
        self.mpc_policy = mpc_policy
        self.target_mpc_policy = target_mpc_policy
        self.target_critic = target_critic
        if self.policy is not None:
            self.policy_optimizer = optim.Adam(self.policy.parameters(), 
                                        lr=float(self.cfg['policy_optimizer']['lr']))
        if self.critic is not None:
            self.critic_optimizer =  optim.Adam(self.critic.parameters(), 
                                        lr=float(self.cfg['critic_optimizer']['lr']))

        self.polyak_tau = float(self.cfg['polyak_tau'])
        self.discount = self.cfg['discount']
        self.num_action_samples = self.cfg['num_action_samples']
        self.num_train_episodes_per_epoch = self.cfg['num_train_episodes_per_epoch']
        self.num_updates_per_epoch = self.cfg.get('num_updates_per_epoch', None)
        self.update_to_data_ratio = self.cfg['update_to_data_ratio']
        self.policy_update_delay = self.cfg['policy_update_delay']
        # self.automatic_entropy_tuning = self.cfg['automatic_entropy_tuning']
        # self.backup_entropy = self.cfg['backup_entropy']
        self.min_buffer_size = int(self.cfg['min_buffer_size'])
        self.reward_scale = self.cfg['reward_scale']
        self.num_eval_episodes = self.cfg.get('num_eval_episodes', 1)
        self.eval_first_policy = self.cfg.get('eval_first_policy', False)
        self.use_mpc_value_targets = self.cfg.get('use_mpc_value_targets', False)
        self.learn_policy = self.cfg.get('learn_policy', True)
        self.policy_use_tanh = self.cfg.get('policy_use_tanh', False)
        self.grad_norm_clip_val = self.cfg.get('grad_norm_clip_val', 0.0)
        self.deterministic_data_collection = self.cfg.get('deterministic_data_collection', False)

    def train(self, debug:bool=False, model_dir=None):
        num_epochs = int(self.cfg['num_epochs'])
        total_env_steps = 0
        log_metrics = defaultdict(list)
        pbar = tqdm(range(int(num_epochs)), desc='train')
        
        for i in pbar:
            #collect new experience
            new_buffer, play_metrics = self.collect_experience(
                policy=self.mpc_policy,
                num_episodes=self.num_train_episodes_per_epoch, 
                update_buffer=True, 
                deterministic=self.deterministic_data_collection,
                debug=debug)

            num_steps_collected = play_metrics['num_steps_collected'] 
            total_env_steps += num_steps_collected

            for k,v in play_metrics.items():
                log_metrics['train/episode/{}'.format(k)].append(v)

            #compute episode metrics
            episode_metric_list = [self.task.compute_metrics(episode) for episode in new_buffer.episode_iterator(
                max_episode_length=self.envs.max_episode_length - 1)]
            episode_metrics = defaultdict(list)
            for k in episode_metric_list[0].keys():
                [episode_metrics[k].append(l[k]) for l in episode_metric_list]
            for k,v in episode_metrics.items():
                log_metrics['train/episode/{}'.format(k)].extend(v)
                        
            self.buffer.concatenate(new_buffer.qlearning_dataset())

            #update agent
            if len(self.buffer) >= self.min_buffer_size:
                if self.num_updates_per_epoch is not None:
                    num_update_steps = int(self.num_updates_per_epoch)
                elif self.update_to_data_ratio is not None:
                    num_update_steps = int(self.update_to_data_ratio * num_steps_collected)
                else:
                    raise ValueError('Either num_updates_per_epoch or update_to_data_ratio must be provided')
                print('Running {} updates'.format(num_update_steps))

                # for k in range(num_update_steps):
                #     with record_function('sample_batch'):
                #         # qlearning_dataset = self.buffer.qlearning_dataset()
                #         batch = self.buffer.sample(self.cfg['train_batch_size']) #, sample_next_state=True)
                #         batch = dict_to_device(batch, self.device)

                #     if self.relabel_data:
                #         with record_function('relabel_data'):
                #             batch = self.preprocess_batch(batch, compute_cost_and_terminals=True)

                #     train_metrics = self.update(batch)
                train_metrics = self.update(num_update_steps)
                pbar.set_postfix(train_metrics)
                
                for k,v in train_metrics.items():
                    log_metrics['train/losses/{}'.format(k)].append(v)

                #Log stuff
                row = {}
                for k, v in log_metrics.items():
                    row[k.split("/")[-1]] = v[-1]
                    if self.tb_writer is not None:                        
                        self.tb_writer.add_scalar(k, v[-1], total_env_steps)
                if self.logger is not None:
                    self.logger.row(row)

        return log_metrics

    # def update(self, batch_dict):
    def update(self, num_update_steps:int):
        # batch_dict = dict_to_device(batch_dict, self.device)
        policy_info_dict, critic_info_dict = {}, {}
        train_metrics = {}
        avg_policy_loss, avg_critic_loss = 0., 0.

        for k in range(num_update_steps):
            with record_function('sample_batch'):
                # qlearning_dataset = self.buffer.qlearning_dataset()
                batch = self.buffer.sample(self.cfg['train_batch_size']) #, sample_next_state=True)
                batch = dict_to_device(batch, self.device)

            if self.relabel_data:
                with record_function('relabel_data'):
                    batch = self.preprocess_batch(batch, compute_cost_and_terminals=True)

            # # batch_dict = dict_to_device(batch_dict, self.device)
            # policy_info_dict, critic_info_dict = {}, {}
            # train_metrics = {}

            batch = self.run_mpc(batch)

            # #Update critic
            # for p1, p2, p3 in zip(self.mpc_policy.policy.controller.value_function.parameters(), self.target_mpc_policy.controller.value_function.parameters(), self.critic.parameters()):
            #     assert torch.allclose(p1, p2) 
            #     assert torch.allclose(p2, p3) 
            # print([p for p in self.mpc_policy.policy.controller.value_function.parameters()])
            # input('...')
            if self.critic is not None:
                self.critic_optimizer.zero_grad()
                # critic_loss, avg_q_value, avg_q_target, max_q_target= self.compute_critic_loss(batch)
                critic_loss, critic_info_dict = self.compute_critic_loss(batch)
                critic_loss.backward()
                if self.grad_norm_clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip_val)
                self.critic_optimizer.step()
                avg_critic_loss += critic_loss.item()

            # for p1, p2 in zip(self.mpc_policy.policy.controller.value_function.parameters(), self.critic.parameters()):
            #     assert torch.allclose(p1, p2) 
            # print([p for p in self.mpc_policy.policy.controller.value_function.parameters()])
            # input('....')

            #Update policy
            if self.learn_policy:
                self.policy_optimizer.zero_grad()
                policy_loss, policy_info_dict = self.compute_policy_loss(batch)
                policy_loss.backward()
                if self.grad_norm_clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm_clip_val)
                self.policy_optimizer.step()
                avg_policy_loss += policy_loss.item()

            #Update target critic using exponential moving average
            if self.critic is not None and self.target_critic is not None:
                for (param1, param2) in zip(self.target_critic.parameters(), self.critic.parameters()):
                    param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)

            # for p1, p2, p3 in zip(self.target_mpc_policy.controller.value_function.parameters(), self.target_critic.parameters(), self.critic.parameters()):
            #     assert torch.allclose(p1, p3) 
            #     assert not torch.allclose(p2, p3)
            # print([p for p in self.target_mpc_policy.controller.value_function.parameters()])
            # input('...')

        train_metrics = {**critic_info_dict, **policy_info_dict}
        train_metrics['avg_qf_loss'] = avg_critic_loss / num_update_steps*1.
        train_metrics['avg_policy_loss'] = avg_policy_loss / num_update_steps*1.
        return train_metrics
    
    def compute_policy_loss(self, batch_dict):
        obs = batch_dict['obs']
        state = batch_dict['state_dict']
        next_state = batch_dict['next_state_dict']
        next_obs = batch_dict['next_obs']
        goal_dict = batch_dict['goal_dict']

        policy_input = {'states': state, 'obs': obs}
        next_policy_input = {'states': next_state, 'obs': next_obs}
    
        # #run mpc to get optimal policy samples
        with torch.no_grad():
        #     reset_data = {}
        #     reset_data['goal_dict'] = copy.deepcopy(goal_dict)
        #     self.target_mpc_policy.reset(reset_data)
        #     optimal_dist, value, info = self.target_mpc_policy(policy_input)
            optimal_dist = batch_dict['optimal_dist']
            act_batch = optimal_dist.sample(torch.Size([self.num_action_samples]))
            next_optimal_dist = batch_dict['next_optimal_dist']
            next_act_batch = next_optimal_dist.sample(torch.Size([self.num_action_samples]))
        
        if self.policy_use_tanh:
            act_batch = torch.tanh(act_batch)
        
        dist = self.policy(policy_input)
        policy_loss_1 = -1.0 * dist.log_prob(act_batch)
        next_dist = self.policy(next_policy_input)
        policy_loss_2 = -1.0 * next_dist.log_prob(next_act_batch)
        policy_loss = torch.cat((policy_loss_1, policy_loss_2), dim=1).mean()
        
        new_actions_1 = dist.sample(torch.Size([self.num_action_samples]))
        log_pi_new_actions_1 = dist.log_prob(new_actions_1) #.mean()
        new_actions_2 = next_dist.sample(torch.Size([self.num_action_samples]))
        log_pi_new_actions_2 = next_dist.log_prob(new_actions_2) #.mean()
        log_pi_new_actions = torch.cat((log_pi_new_actions_1, log_pi_new_actions_2), dim=1).mean()
        action_diff_1 = torch.norm(new_actions_1 - act_batch, dim=-1) #.mean()
        action_diff_2 = torch.norm(new_actions_2 - next_act_batch, dim=-1) #.mean()
        action_diff = torch.cat((action_diff_1, action_diff_2), dim=1).mean()

        policy_info_dict = {
            # 'policy_loss': policy_loss.item(),
            'action_difference': action_diff.item(),
            'policy_entropy': log_pi_new_actions.item()
        }


        # policy_loss = -1.0 * self.policy.log_prob(policy_input, act_batch).mean()
        # new_actions, log_pi_new_actions = self.policy.entropy(policy_input)

        return policy_loss, policy_info_dict #log_pi_new_actions.mean()


    def compute_critic_loss(self, batch_dict):
        cost_batch = batch_dict['cost'].squeeze(-1)
        obs_batch = batch_dict['obs']
        act_batch = batch_dict['actions']
        next_obs_batch = batch_dict['next_obs']
        next_state_batch = batch_dict['next_state_dict']
        done_batch = batch_dict['terminals']

        with torch.no_grad():
            policy_input = {
                'states': next_state_batch,
                'obs': next_obs_batch}
            
            if not self.use_mpc_value_targets:
                # next_actions, next_actions_log_prob = self.policy.entropy(policy_input)
                if self.learn_policy:
                    next_actions_dist = self.policy(policy_input)
                else:
                    # self.target_mpc_policy.reset(dict(goal_dict=goal_dict))
                    # next_actions_dist, _, _ = self.target_mpc_policy(policy_input)
                    next_actions_dist = batch_dict['next_optimal_dist']
                
                next_actions = next_actions_dist.sample(torch.Size([self.num_action_samples]))
                next_actions_log_prob = next_actions_dist.log_prob(next_actions)
                next_actions_log_prob = next_actions_log_prob.mean(-1) #mean along action dimension

                q_pred_next = self.target_critic(
                    {'obs': next_obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
                    next_actions).mean(0)
                # q_pred_next = target_pred.mean(0) #mean across num action samples

            else:
                q_pred_next = batch_dict['next_mpc_value_targets']
                # self.target_mpc_policy.reset(dict(goal_dict=goal_dict))
                # next_actions_dist, q_pred_next, aux_info = self.target_mpc_policy(policy_input)

            # if self.backup_entropy:
            #     alpha = self.log_alpha.exp()
            #     q_pred_next =  target_pred + alpha * next_actions_log_prob # sign is flipped in entropy since we are minimizing costs
            # else:
            #     q_pred_next = target_pred

            q_target = self.reward_scale * cost_batch +  self.discount * q_pred_next #(1. - done_batch) * 

        qf_all = self.critic.all({'obs': obs_batch}, act_batch)
        q_target = q_target.unsqueeze(-1).repeat(1, qf_all.shape[-1])
        
        qf_loss = F.mse_loss(qf_all, q_target, reduction='none')

        qf_loss = qf_loss.sum(-1).mean(0) #sum along ensemble dimension and mean along batch

        avg_q_value = torch.max(qf_all, dim=-1)[0].mean() #max instead of min since we are minimizing costs
        avg_target_value = q_target.mean()
        max_target_value = q_target.max()

        qf_info_dict = {
            # 'qf_loss': qf_loss.item(),
            'avg_q_value': avg_q_value.item(),
            'avg_target_value': avg_target_value.item(),
            'max_target_value': max_target_value.item(),

        }

        return qf_loss, qf_info_dict


    def run_mpc(self, batch_dict):
        obs = batch_dict['obs']
        state = batch_dict['state_dict']
        goal_dict = batch_dict['goal_dict']
        next_state = batch_dict['next_state_dict']
        next_obs = batch_dict['next_obs']

        #run mpc to get optimal policy samples
        with torch.no_grad():
            state.pop("prev_action", None)
            policy_input = {
                'states': state,
                'obs': obs}
            reset_data = {}
            reset_data['goal_dict'] = copy.deepcopy(goal_dict)
            self.target_mpc_policy.reset(reset_data)
            optimal_dist, value_preds, info = self.target_mpc_policy(policy_input, calc_val=True)

            #also run for next state
            next_state.pop("prev_action", None)
            policy_input = {
                'states': next_state,
                'obs': next_obs}

            self.target_mpc_policy.reset(reset_data)
            next_optimal_dist, next_value_preds, info = self.target_mpc_policy(policy_input, calc_val=True)

        batch_dict['optimal_dist'] = optimal_dist
        batch_dict['mpc_value_targets'] = value_preds
        batch_dict['next_optimal_dist'] = next_optimal_dist 
        batch_dict['next_mpc_value_targets'] = next_value_preds 

        return batch_dict