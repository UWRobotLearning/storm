import copy
from collections import defaultdict
from typing import Optional, Dict
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from storm_kit.learning.agents import Agent
from storm_kit.learning.learning_utils import dict_to_device, asymmetric_l2_loss, update_exponential_moving_average
from storm_kit.learning.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

class BPAgent(nn.Module):
    def __init__(
        self,
        cfg,
        policy=None,
        qf=None,
        vf=None,
        target_policy=None,
        target_qf=None,
        target_vf=None,
        device=torch.device('cpu'), 
    ):
        super().__init__()
        self.cfg = cfg
        self.policy = policy
        self.qf = qf
        self.vf = vf
        self.device = device 

        if self.policy is not None:
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg['policy_optimizer']['lr'])
            self.target_policy = copy.deepcopy(self.policy).requires_grad_(False) if target_policy is None else target_policy

        if self.qf is not None:
            self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=self.cfg['qf_optimizer']['lr'])
            self.target_qf = copy.deepcopy(self.qf).requires_grad_(False) if target_qf is None else target_qf

        if self.vf is not None:
            self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.cfg['vf_optimizer']['lr'])
            self.target_vf = copy.deepcopy(self.vf).requires_grad_(False) if target_vf is None else target_vf
        
        self.setup_agent_called = False
 
        self.num_action_samples = self.cfg['num_action_samples']
        self.fixed_alpha = self.cfg.get('fixed_alpha', 0.2)
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
        self.randomize_ensemble_targets = self.cfg.get('randomize_ensemble_targets', True)

    def update(
            self, batch:Dict[str, torch.Tensor], 
            step_num:int, normalization_stats:Dict[str, float] = {'disc_return_mean': 0.0, 'disc_return_std': 1.0}, 
            debug:bool = False):
        
        assert self.setup_agent_called, "Agent setup must be called before training"
        
        policy_info_dict, qf_info_dict, vf_info_dict = {}, {}, {}

        #Update value function
        if self.vf is not None:
            # for (param1, param2) in zip(self.target_vf.parameters(), self.vf.parameters()):
            #     param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)
            update_exponential_moving_average(self.target_vf, self.vf, self.polyak_tau)
            vf_loss, adv, vf_info_dict = self.compute_vf_loss(batch, step_num, normalization_stats)
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()
            batch['adv'] = adv

        #Update Q function
        if self.qf is not None:
            # for (param1, param2) in zip(self.target_qf.parameters(), self.qf.parameters()):
            #     param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)
            update_exponential_moving_average(self.target_qf, self.qf, self.polyak_tau)
            qf_loss, qf_info_dict = self.compute_qf_loss(batch, step_num, normalization_stats)
            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()
        
        #Update policy
        if self.policy is not None:
            update_exponential_moving_average(self.target_policy, self.policy, self.polyak_tau)
            policy_loss, policy_info_dict = self.compute_policy_loss(batch, step_num, normalization_stats)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            self.policy_lr_schedule.step()
            
        info_dict = {**policy_info_dict, **qf_info_dict, **vf_info_dict}
        
        return info_dict


    def compute_policy_loss(
            self, batch:Dict[str, torch.Tensor], 
            step_num:int, normalization_stats:Dict[str, float] = {'disc_return_mean': 0.0, 'disc_return_std': 1.0}):
        
        obs_batch = batch['observations']
        act_batch = batch['actions']
        
        if self.policy_use_tanh:
            act_batch = torch.tanh(act_batch)

        policy_out = self.policy({'obs': obs_batch})

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
        
        # if step_num < self.num_warmstart_steps:
        policy_loss = bc_loss.mean()
        # else:
        #     #compute advantage weighted behavior cloning loss
        #     if self.vf is not None and self.vf_target_mode=='asymmetric_l2':
        #         #IQL loss
        #         if 'adv' in batch: adv = batch['adv']
        #         else:
        #             v_target = self.target_qf(
        #                 {'obs': obs_batch}, act_batch)
        #             v_pred = self.vf({'obs': obs_batch})  # inference
        #             adv = v_target - v_pred
        #         weight = torch.exp(self.beta * adv.detach()).clamp(max=100.)
        #     else:
        #         #CRR loss
        #         q_pred = self.qf(
        #             {'obs': obs_batch}, act_batch)
        #         action_samples = torch.stack([policy_out.sample() for _ in range(self.num_adv_samples)], dim=0)
        #         repeat_obs = torch.repeat_interleave(obs_batch.unsqueeze(0), self.num_adv_samples, 0)
        #         v_pred = self.qf(
        #             {'obs': repeat_obs}, action_samples)
        #         if self.advantage_mode == 'mean':
        #             advantage = q_pred - v_pred.mean(dim=0)
        #         elif self.advantage_mode == 'max':
        #             advantage = q_pred - v_pred.max(dim=0)[0]
                
        #         if self.weight_mode == 'exp':
        #             weight = torch.exp(advantage / self.beta)
        #         elif self.weight_mode == 'binary':
        #             weight = (advantage > 0).float()
        #         weight = torch.clamp_max(weight, 20).detach()
            
        #     policy_loss = torch.mean(weight * bc_loss.mean(-1))

        policy_info_dict = {
            'Train/bc_loss': bc_loss.mean().item(),
            'Train/policy_loss': policy_loss.item(),
            'Train/action_difference': action_diff.item(),
            'Train/policy_entropy': log_pi_new_actions.item()
        }

        return policy_loss, policy_info_dict
    
    def compute_qf_loss(
            self, batch_dict:Dict[str, torch.Tensor], 
            step_num:int, normalization_stats:Dict[str, float] = {'disc_return_mean': 0.0, 'disc_return_std': 1.0}):
        
        assert self.policy is not None, "Q-function learning requires policy"
        
        r_c_batch = batch_dict['costs'] if 'costs' in batch_dict else batch_dict['rewards']
        obs_batch = batch_dict['observations']
        act_batch = batch_dict['actions']
        next_obs_batch = batch_dict['next_observations']
        next_state_batch = batch_dict['next_state_dict']
        done_batch = batch_dict['terminals'].float()

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

    def compute_vf_loss(
            self, batch:Dict[str, torch.Tensor], 
            step_num:int, normalization_stats:Dict[str, float] = {'disc_return_mean': 0.0, 'disc_return_std': 1.0, 'obs_mean': None, 'obs_std': None}):
        
        r_c_batch = batch['costs'] if 'costs' in batch else batch['rewards']
        obs_batch = batch['observations']
        next_obs_batch = batch['next_observations']
        done_batch = batch['terminals'].float()
        last_terminals = batch['last_terminals'].float()
        last_observations = batch['last_observations']
        returns = batch['disc_returns']
        remaining_steps = batch['remaining_steps']

        mc_targets = torch.zeros_like(r_c_batch)
        td_targets = torch.zeros_like(r_c_batch)
        with torch.no_grad():
            if self.lambd > 0:
                last_inp = self.get_normalized_input(last_observations, normalization_stats)
                last_vs, _ = self.vf.all(last_inp)  # inference (normalized space)
                # last_vs = normalization_stats['disc_return_std'] * last_vs + normalization_stats['disc_return_mean'] #un-normalize
                last_vs = self.unnormalize_value_estimates(last_vs, normalization_stats)
                # last_vs += normalization_stats['disc_return_mean'] #un-normalize
                mc_targets = (returns + (1. - last_terminals) * (self.discount**remaining_steps) * last_vs)#.clamp(min=self.V_min, max=self.V_max)

            if self.lambd < 1:
                next_inp = self.get_normalized_input(next_obs_batch, normalization_stats)    
                v_next, _ = self.vf.all(next_inp) #inference (normalized space)
                # v_next = normalization_stats['disc_return_std'] * v_next + normalization_stats['disc_return_mean'] #un-normalize         
                v_next = self.unnormalize_value_estimates(v_next, normalization_stats) #un-normalize         
                # v_next += normalization_stats['disc_return_mean'] #un-normalize         
                td_targets = (r_c_batch +  (1. - done_batch) * self.discount * v_next)#.clamp(min=self.V_min, max=self.V_max)
        
        v_target = td_targets*(1.-self.lambd) + mc_targets*self.lambd
        # adv = v_target - v_pred
        
        inp = self.get_normalized_input(obs_batch, normalization_stats) 
        vf_all, vf_pred_info = self.vf.all(inp) #inference (normalized space)
        # v_target = v_target.unsqueeze(0).repeat(vf_all.shape[0],1)                
        # v_target = (v_target - disc_return_mean) # / disc_return_std  #Normalize targets 
        v_target = self.normalize_value_estimates(v_target, normalization_stats)



        vf_loss = F.mse_loss(vf_all, v_target, reduction='none') #loss (normalized space)
        
        if self.randomize_ensemble_targets:
            #create random mask 
            # x = torch.randn_like(v_target)
            # mask = x.ge(0.5)
            # non_zero = mask.sum(-1)
            vf_loss = vf_loss * self.mask.float()
            vf_loss = vf_loss.sum(-1)
            vf_loss = (vf_loss / self.non_zero) #mean along batch
            vf_loss = vf_loss.sum(0) #sum along ensemble dimension
        else:
            vf_loss = vf_loss.sum(0).mean() #sum along ensemble dimension and mean along batch

        # avg_v_value = torch.max(vf_all, dim=0)[0].mean() 
        avg_target_value = v_target.mean()
        max_target_value = v_target.max()
        min_target_value = v_target.min()

        vf_info_dict = {
            'Train/vf_loss': vf_loss.item(),
            'Train/vf_ensemble_mean': vf_pred_info['mean'].mean().item(),
            'Train/vf_ensemble_std': vf_pred_info['std'].mean().item(),
            'Train/avg_vtarget_value': avg_target_value.item(),
            'Train/max_vtarget_value': max_target_value.item(),
            'Train/min_vtarget_value': min_target_value.item(),
        }        
        
        return vf_loss, None, vf_info_dict
        
    def get_normalized_input(self, obs, normalization_stats):
        # obs_mean, obs_std = normalization_stats['obs_mean'], normalization_stats['obs_std']
        # if obs_mean is not None:
        #     obs -= obs_mean
        # # if obs_std is not None:
        # #     obs /= obs_std

        obs_max, obs_min = normalization_stats['obs_max'], normalization_stats['obs_min']
        obs_range = (obs_max - obs_min) + 1e-12
        obs = (obs - obs_min) / obs_range 
        # if obs_std is not None:
        #     obs /= obs_std
        return {'obs': obs}

    def normalize_value_estimates(self, values, normalization_stats):
        V_min, V_max = normalization_stats['V_min'], normalization_stats['V_max']
        V_range = (V_max - V_min) + 1e-12
        values = (values - V_min) / V_range
        return values

    def unnormalize_value_estimates(self, values, normalization_stats):
        V_min, V_max = normalization_stats['V_min'], normalization_stats['V_max']
        V_range = V_max - V_min
        values =  values * V_range + V_min 
        return values

    def state_dict(self):
        state = {}
        if self.policy is not None:
            state['policy_state_dict'] = self.policy.state_dict()
        if self.qf is not None:
            state['qf_state_dict'] = self.qf.state_dict()
        if self.vf is not None:
            state['vf_state_dict'] = self.vf.state_dict()

        return state
 
    def setup(self, max_steps:int, batch_size:int, ensemble_size:int=1):
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        if ensemble_size > 1:
            x = torch.randn(ensemble_size, batch_size, device=self.device)
            self.mask = x.ge(0.5)
        else:
            self.mask = torch.ones(ensemble_size, batch_size, device=self.device).bool()
        self.non_zero = self.mask.sum(-1)
        self.setup_agent_called = True
 
    def compute_validation_metrics(
            self, 
            validation_buffer:ReplayBuffer, 
            success_buffer:Optional[ReplayBuffer]=None, 
            normalization_stats:Dict={'disc_return_mean': 0.0, 'disc_return_std': 1.0}):
        
        info = {}
        with torch.no_grad():
            obs = validation_buffer['observations'].clone()
            inp = self.get_normalized_input(obs, normalization_stats)
            vf_preds, vf_preds_info = self.vf.all(inp) #inference (normalized)
            # vf_preds = normalization_stats['disc_return_std'] * vf_preds + normalization_stats['disc_return_mean'] #un-normalize
            # vf_preds += normalization_stats['disc_return_mean'] #un-normalize
            vf_preds = self.unnormalize_value_estimates(vf_preds, normalization_stats) #un-normalize

            disc_returns = validation_buffer['disc_returns'].unsqueeze(0).repeat(vf_preds.shape[0], 1) #ensemble x num_points
            vf_error_validation = torch.abs(vf_preds - disc_returns).mean().item() #sum across trajectory and moean along ensemble
            num_members = vf_preds.shape[0]
            for i in range(num_members):
                validation_buffer[f'vf_preds_{i}'] = vf_preds[i,:]

            #plot predictions for each trajectory
            # fig_list, ax_list = [], []
            figs = {}
            for episode_num, episode in enumerate(validation_buffer.episode_iterator()):
                fig, ax = plt.subplots()
                ax.plot(episode['disc_returns'].cpu().numpy(), linestyle='solid')
                for ensemble_idx in range(num_members):
                    vf_preds_i = episode[f'vf_preds_{ensemble_idx}'].cpu().numpy()
                    ax.plot(vf_preds_i, linestyle='dashed', label=f'ensemble_mem_{ensemble_idx}')
                if episode_num == 0:
                    ax.legend()
                figs[f'{episode_num}'] = fig
                # fig_list.append(fig)
                # ax_list.append(ax)

        info['Train/vf_error_validation'] = vf_error_validation
        info['Train/vf_ensemble_mean_validation'] = vf_preds_info['mean'].mean().item()
        info['Train/vf_ensemble_std_validation'] = vf_preds_info['std'].mean().item()

        return info, figs


        # if success_buffer is not None:
        #     with torch.no_grad():
        #         disc_returns_success = success_batch['disc_returns']
        #         vf_success = self.vf({'obs': success_batch['observations']})
        #         vf_success = disc_return_std * vf_success + disc_return_mean #unnormalize
        #         vf_error_success = torch.abs(vf_success - disc_returns_success).mean()#torch.norm(vf_success - disc_returns_success, p=2, dim=-1)

        #         disc_returns_batch = batch['disc_returns']
        #         vf_batch = self.vf({'obs': batch['observations']})
        #         vf_batch = disc_return_std * vf_batch + disc_return_mean #unnormalize
        #         vf_error_batch = torch.abs(vf_batch - disc_returns_batch).mean() #torch.norm(vf_batch - disc_returns_batch, p=2, dim=-1)

        #     vf_info_dict['Train/vf_error_success'] = vf_error_success.item()
        #     vf_info_dict['Train/vf_error_batch'] = vf_error_batch.item()        
            
            # v_pred = self.vf({'obs': obs_batch})  # inference        
    # if self.vf_target_mode == 'asymmetric_l2':
        #     with torch.no_grad():
        #         v_target = self.target_qf(
        #             {'obs': obs_batch}, actions
        #         ).clamp(min=self.V_min, max=self.V_max)
        #     adv = v_target - v_pred
        #     vf_loss = asymmetric_l2_loss(adv, self.expecile_tau) 
        # else: