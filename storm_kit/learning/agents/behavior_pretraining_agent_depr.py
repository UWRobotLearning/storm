import copy
from collections import defaultdict
from typing import Optional, Dict
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn.functional as F
from torch.profiler import record_function

from storm_kit.learning.agents import Agent
from storm_kit.learning.learning_utils import dict_to_device, asymmetric_l2_loss
from storm_kit.mpc.control.control_utils import cost_to_go
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
        runner_fn,
        # mpc_policy,
        policy=None,
        qf=None,
        vf=None,
        target_qf=None,
        target_vf=None,
        max_steps=np.inf,
        V_min=-float('inf'),
        V_max=float('inf'),
        logger=None,
        tb_writer=None,
        device=torch.device('cpu'), 
        eval_rng: Optional[torch.Generator]=None
    ):
        super().__init__(
            cfg, envs, task, obs_dim, action_dim, #obs_space, action_space,
            buffer=buffer, policy=policy,
            runner_fn=runner_fn,
            logger=logger, tb_writer=tb_writer,
            device=device, eval_rng=eval_rng
        )

        self.qf = qf
        self.vf = vf 

        if self.policy is not None:
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg['policy_optimizer']['lr'])
            self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)

        if self.qf is not None:
            self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=self.cfg['qf_optimizer']['lr'])

        if self.vf is not None:
            self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.cfg['vf_optimizer']['lr'])

        if self.qf is not None:
            # assert self.policy is not None, 'Learning a q network requires a policy network.'
            self.target_qf = copy.deepcopy(self.qf).requires_grad_(False) if target_qf is None else target_qf

        if self.vf is not None:
            self.target_vf = copy.deepcopy(self.vf).requires_grad_(False) if target_vf is None else target_vf
 
        self.V_min, self.V_max = V_min, V_max
        self.num_action_samples = self.cfg['num_action_samples']
        # self.fixed_alpha = self.cfg['fixed_alpha']
        # if self.policy_loss_type not in ["mse", "nll"]:
        #     raise ValueError('Unidentified policy loss type {}.'.format(self.policy_loss_type))
        self.num_eval_episodes = self.cfg.get('num_eval_episodes', 1)
        self.eval_first_policy = self.cfg.get('eval_first_policy', False)
        self.policy_use_tanh = self.cfg.get('policy_use_tanh', False)
        self.discount = self.cfg.get('discount')
        self.polyak_tau = float(self.cfg['polyak_tau'])
        self.num_warmstart_steps = self.cfg.get('num_warmstart_steps', np.inf)
        self.beta = self.cfg.get('beta', 1.0)
        self.num_adv_samples = self.cfg.get('num_adv_samples', 4)
        self.advantage_mode = self.cfg.get('advantage_mode', 'max')
        self.weight_mode = self.cfg.get('weight_mode', 'exp')
        self.vf_target_mode = self.cfg.get('vf_target_mode', 'asymmetric_l2')
        self.expecile_tau = self.cfg.get('expecile_tau', 0.7)
        self.lambd = self.cfg.get('lambd', 1.0)

    def train(self, model_dir=None, data_dir=None, debug:bool=False):
        num_train_steps = self.cfg['num_pretrain_steps']
        log_metrics = defaultdict(list)
        pbar = tqdm(range(int(num_train_steps)), desc='train')

        for i in pbar:
            #Evaluate policy at some frequency
            if ((i + (1-self.eval_first_policy)) % self.eval_freq == 0) or (i == num_train_steps -1):
                print('[BehaviorPretraining]: Evaluating policy')
                self.policy.eval()
                eval_buffer, eval_metrics = self.evaluate_policy(
                    self.policy, 
                    num_eval_episodes=self.num_eval_episodes, 
                    deterministic=True, 
                    debug=False)
               
                print(eval_metrics)
                for k,v in eval_metrics.items():
                    log_metrics['eval/episode/{}'.format(k)].append(v)
                episode_metric_list = [self.task.compute_metrics(episode) for episode in eval_buffer.episode_iterator(
                    max_episode_length=self.envs.max_episode_length - 1)]
                episode_metrics = defaultdict(list)
                for k in episode_metric_list[0].keys():
                    [episode_metrics[k].append(l[k]) for l in episode_metric_list]
                for k,v in episode_metrics.items():
                    log_metrics['eval/episode/{}'.format(k)].extend(v)

                self.policy.train()
                pbar.set_postfix(eval_metrics)

            with record_function('sample_batch'):
                batch = self.buffer.sample(self.cfg['train_batch_size']) #, sample_next_state=False)
                batch = dict_to_device(batch, self.device)
            
            if self.relabel_data:
                with record_function('relabel_data'):
                    batch = self.preprocess_batch(batch, compute_cost_and_terminals=True)
            
            with record_function('update'):
                train_metrics = self.update(batch, i)
            pbar.set_postfix(train_metrics)

            for k,v in train_metrics.items():
                log_metrics['train/losses/{}'.format(k)].append(v)

            #Log stuff
            row = {}
            for k, v in log_metrics.items():
                row[k.split("/")[-1]] = v[-1]
                if self.tb_writer is not None:                        
                    self.tb_writer.add_scalar(k, v[-1], i)
            if self.logger is not None:
                self.logger.row(row)

            # if self.tb_writer is not None:
            #     for k, v in train_metrics.items():
            #         self.tb_writer.add_scalar('Train/' + k, v, i)
                        
            if (i % self.checkpoint_freq == 0) or (i == num_train_steps -1):
                print(f'Iter {i}: Saving current policy')
                self.save(model_dir, data_dir, iter=0)
            

    def update(self, batch:Dict[str, torch.Tensor], success_batch:Dict[str, torch.Tensor], step_num:int, debug:bool = False):

        policy_info_dict, qf_info_dict, vf_info_dict = {}, {}, {}

        #Update value function
        if self.vf is not None:
            #Update target vf using exponential moving average
            for (param1, param2) in zip(self.target_vf.parameters(), self.vf.parameters()):
                param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)
            vf_loss, adv, vf_info_dict = self.compute_vf_loss(batch, success_batch, step_num, debug)

            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()
            batch['adv'] = adv

        #Update q function
        if self.qf is not None:
            qf_loss, qf_info_dict = self.compute_qf_loss(batch, success_batch, step_num, debug)
            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()
        
        #Update policy
        if self.policy is not None:
            policy_loss, policy_info_dict = self.compute_policy_loss(batch, success_batch, step_num, debug)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            self.policy_lr_schedule.step()

        # for p1, p2 in zip(self.mpc_policy.policy.controller.sampling_policy.parameters(), self.policy.parameters()):
        #     assert torch.allclose(p2, p2)
        # print([p for p in self.mpc_policy.policy.controller.sampling_policy.parameters()])
        # input('...')
        # for p1, p2 in zip(self.mpc_policy.policy.controller.value_function.parameters(), self.qf.parameters()):
        #     assert torch.allclose(p1, p2)
        # print([p for p in self.mpc_policy.policy.controller.value_function.parameters()])
        # input('...')
            
        info_dict = {**policy_info_dict, **qf_info_dict, **vf_info_dict}
        
        return info_dict


    def compute_policy_loss(self, batch:Dict[str, torch.Tensor], success_batch:Dict[str, torch.Tensor], step_num:int, debug:bool=False):
        obs_batch = batch['observations']
        state_batch = batch['state_dict']
        act_batch = batch['actions']
        
        if self.policy_use_tanh:
            act_batch = torch.tanh(act_batch)

        policy_input = {
            'states': state_batch,
            'obs': obs_batch}

        policy_out = self.policy(policy_input)

        if isinstance(policy_out, torch.distributions.Distribution):  # MLE
            bc_loss = -1.0 * policy_out.log_prob(act_batch)
            new_actions = policy_out.sample()
            log_pi_new_actions = policy_out.log_prob(new_actions).mean()
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == act_batch.shape
            new_actions = policy_out
            bc_loss = F.mse_loss(new_actions, act_batch, reduction=None)
            log_pi_new_actions = torch.tensor([0.0])

        action_diff = torch.norm(new_actions - act_batch, dim=-1).mean()
        
        if step_num < self.num_warmstart_steps:
            policy_loss = bc_loss.mean()
        else:
            #compute advantage weighted behavior cloning loss
            if self.vf is not None and self.vf_target_mode=='asymmetric_l2':
                #IQL loss
                if 'adv' in batch: adv = batch['adv']
                else:
                    v_target = self.target_qf(
                        {'obs': obs_batch}, act_batch)
                    v_pred = self.vf({'obs': obs_batch})  # inference
                    adv = v_target - v_pred
                weight = torch.exp(self.beta * adv.detach()).clamp(max=100.)
            else:
                #CRR loss
                q_pred = self.qf(
                    {'obs': obs_batch}, act_batch)
                action_samples = torch.stack([policy_out.sample() for _ in range(self.num_adv_samples)], dim=0)
                repeat_obs = torch.repeat_interleave(obs_batch.unsqueeze(0), self.num_adv_samples, 0)
                v_pred = self.qf(
                    {'obs': repeat_obs}, action_samples)
                if self.advantage_mode == 'mean':
                    advantage = q_pred - v_pred.mean(dim=0)
                elif self.advantage_mode == 'max':
                    advantage = q_pred - v_pred.max(dim=0)[0]
                
                if self.weight_mode == 'exp':
                    weight = torch.exp(advantage / self.beta)
                elif self.weight_mode == 'binary':
                    weight = (advantage > 0).float()
                weight = torch.clamp_max(weight, 20).detach()
            
            policy_loss = torch.mean(weight * bc_loss.mean(-1))


        policy_info_dict = {
            'Train/bc_loss': bc_loss.mean().item(),
            'Train/policy_loss': policy_loss.item(),
            'Train/action_difference': action_diff.item(),
            'Train/policy_entropy': log_pi_new_actions.item()
        }

        return policy_loss, policy_info_dict
    
    def compute_qf_loss(self, batch_dict:Dict[str, torch.Tensor], success_batch:Dict[str, torch.Tensor], step_num:int, debug:bool=False):
        assert self.policy is not None, "Q-function learning requires policy"
        r_c_batch = batch_dict['costs'] if 'costs' in batch_dict else batch_dict['rewards']
        obs_batch = batch_dict['observations']
        act_batch = batch_dict['actions']
        next_obs_batch = batch_dict['next_observations']
        next_state_batch = batch_dict['next_state_dict']
        done_batch = batch_dict['terminals'].float()
        #Update target critic using exponential moving average
        for (param1, param2) in zip(self.target_qf.parameters(), self.qf.parameters()):
            param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)

        with torch.no_grad():
            if self.vf is not None and self.vf_target_mode == 'asymmetric_l2':
                #use vf to compute q-target similar to IQL
                v_next = self.vf({'obs': next_obs_batch})

            else:
                #query policy to compute q target
                policy_input = {
                    'states': next_state_batch,
                    'obs': next_obs_batch}
                
                next_actions_dist = self.policy(policy_input)                
                next_actions = next_actions_dist.sample(torch.Size([self.num_action_samples]))

                v_next = self.target_qf(
                    {'obs': next_obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
                    next_actions).mean(0)
            
            q_target = (r_c_batch +  (1. - done_batch) * self.discount * v_next).clamp(min=self.V_min, max=self.V_max)

        qf_all = self.qf.all({'obs': obs_batch}, act_batch)
        q_target = q_target.unsqueeze(-1).repeat(1, qf_all.shape[-1])

        qf_loss = F.mse_loss(qf_all, q_target, reduction='none')
        qf_loss = qf_loss.sum(-1).mean(0) #sum along ensemble dimension and mean along batch

        avg_q_value = torch.min(qf_all, dim=-1)[0].mean() 
        avg_target_value = q_target.mean()
        max_target_value = q_target.max()
        min_target_value = q_target.min()
       
        qf_info_dict = {
            'Train/qf_loss': qf_loss.item(),
            'Train/avg_q_value': avg_q_value.item(),
            'Train/avg_target_value': avg_target_value.item(),
            'Train/max_target_value': max_target_value.item(),
            'Train/min_target_value': min_target_value.item(),
        }
        return qf_loss, qf_info_dict

    def compute_vf_loss(self, batch:Dict[str, torch.Tensor], success_batch:Dict[str,torch.Tensor], step_num:int, debug:bool=False):
        
        r_c_batch = batch['costs'] if 'costs' in batch else batch['rewards']
        obs_batch = batch['observations']
        next_obs_batch = batch['next_observations']
        done_batch = batch['terminals'].float()
        last_terminals = batch['last_terminals'].float()
        last_observations = batch['last_observations']
        returns = batch['disc_returns']
        remaining_steps = batch['remaining_steps']

        disc_return_mean = batch['disc_return_mean'] if 'disc_return_mean' in batch else 0.0
        disc_return_std = (batch['disc_return_std'] + 1e-12) if 'disc_return_std' in batch else 1.0

        mc_targets = torch.zeros_like(r_c_batch)
        td_targets = torch.zeros_like(r_c_batch)
        with torch.no_grad():
            if self.lambd > 0:
                last_vs = self.target_vf({'obs': last_observations})  # inference
                mc_targets = (returns + (1. - last_terminals) * (self.discount**remaining_steps) * last_vs)#.clamp(min=self.V_min, max=self.V_max)

            if self.lambd < 1:    
                v_next = self.target_vf({'obs': next_obs_batch})            
                td_targets = (r_c_batch +  (1. - done_batch) * self.discount * v_next)#.clamp(min=self.V_min, max=self.V_max)
        v_target = td_targets*(1.-self.lambd) + mc_targets*self.lambd
        # adv = v_target - v_pred
        # vf_loss = F.mse_loss(v_pred, v_target, reduction='none').mean()
        
        vf_all = self.vf.all({'obs': obs_batch})
        v_target = v_target.unsqueeze(0).repeat(vf_all.shape[0],1)
        #Normalize targets 
        v_target = (v_target - disc_return_mean) / disc_return_std

        vf_loss = F.mse_loss(vf_all, v_target, reduction='none')
        vf_loss = vf_loss.sum(0).mean() #sum along ensemble dimension and mean along batch

        avg_v_value = torch.max(vf_all, dim=0)[0].mean() 
        avg_target_value = v_target.mean()
        max_target_value = v_target.max()
        min_target_value = v_target.min()

        vf_info_dict = {
            'Train/vf_loss': vf_loss.item(),
            'Train/avg_v_value': avg_v_value.item(),
            'Train/avg_vtarget_value': avg_target_value.item(),
            'Train/max_vtarget_value': max_target_value.item(),
            'Train/min_vtarget_value': min_target_value.item(),
        }        
        
        if debug:
            if success_batch is not None:
                with torch.no_grad():
                    disc_returns_success = success_batch['disc_returns']
                    vf_success = self.vf({'obs': success_batch['observations']})
                    vf_success = disc_return_std * vf_success + disc_return_mean #unnormalize
                    vf_error_success = torch.abs(vf_success - disc_returns_success).mean()#torch.norm(vf_success - disc_returns_success, p=2, dim=-1)

                    disc_returns_batch = batch['disc_returns']
                    vf_batch = self.vf({'obs': batch['observations']})
                    vf_batch = disc_return_std * vf_batch + disc_return_mean #unnormalize
                    vf_error_batch = torch.abs(vf_batch - disc_returns_batch).mean() #torch.norm(vf_batch - disc_returns_batch, p=2, dim=-1)

                    # r_c_batch_succ = success_batch['costs'] if 'costs' in success_batch else success_batch['rewards']
                    # v_next_success = self.target_vf({'obs': success_batch['next_observations']})            
                    # td_targets_success = (r_c_batch_succ +  (1. - success_batch['terminals'].float()) * self.discount * v_next_success)
                    # vf_all_success = self.vf.all({'obs': success_batch['observations']})
                    # v_target_success = td_targets_success.unsqueeze(0).repeat(vf_all_success.shape[0],1)

                    # vf_loss_success = F.mse_loss(vf_all_success, v_target_success, reduction='none')
                    # vf_loss_success = vf_loss_success.sum(0).mean() #sum along ensemble dimension and mean along batch
                vf_info_dict['Train/vf_error_success'] = vf_error_success.item()
                vf_info_dict['Train/vf_error_batch'] = vf_error_batch.item()
        
        return vf_loss, None, vf_info_dict
        



        # v_pred = self.vf({'obs': obs_batch})  # inference        
        # if self.vf_target_mode == 'asymmetric_l2':
        #     with torch.no_grad():
        #         v_target = self.target_qf(
        #             {'obs': obs_batch}, actions
        #         ).clamp(min=self.V_min, max=self.V_max)
        #     adv = v_target - v_pred
        #     vf_loss = asymmetric_l2_loss(adv, self.expecile_tau) 
        # else: