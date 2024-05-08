#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#

import copy

import numpy as np
import scipy.special
import torch
# from torch.nn.functional import normalize as f_norm
from torch.profiler import record_function

from .control_utils import cost_to_go, matrix_cholesky
from .gaussian_mpc import GaussianMPC

class MPPI(GaussianMPC):
    """
    .. inheritance-diagram:: MPPI
       :parts: 1

    Class that implements Model Predictive Path Integral Controller
    
    Implementation is based on 
    Williams et. al, Information Theoretic MPC for Model-Based Reinforcement Learning
    with additional functions for updating the covariance matrix
    and calculating the soft-value function.

    """

    def __init__(self,
                 d_action,
                 horizon,
                 init_cov,
                 init_mean,
                 base_action,
                 beta,
                 num_particles,
                 step_size_mean,
                 step_size_cov,
                 alpha,
                 gamma,
                 td_lam,
                 kappa,
                 n_iters,
                 action_lows,
                 action_highs,
                 null_act_frac:float=0.,
                 cl_act_frac:float=0.,
                 use_cl_std:bool = False,
                #  rollout_fn=None,
                 task=None,
                 dynamics_model=None,
                 sampling_policy=None,
                #  value_function=None,
                 vf=None,
                 qf=None,
                 hotstart=True,
                 state_batch_size:int = 1,
                #  num_instances=1,
                 squash_fn='clamp',
                 update_cov=False,
                 cov_type='sigma_I',
                 seed=0,
                 sample_params={'type': 'halton', 'fixed_samples': True, 'seed':0, 'filter_coeffs':None},
                 normalize_returns=False,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32},
                 visual_traj='state_seq'):
        
        super(MPPI, self).__init__(d_action,
                                   action_lows, 
                                   action_highs,
                                   horizon,
                                   init_cov,
                                   init_mean,
                                   base_action,
                                   num_particles,
                                   gamma,
                                   td_lam,
                                   n_iters,
                                   step_size_mean,
                                   step_size_cov, 
                                   null_act_frac,
                                   cl_act_frac,
                                   use_cl_std,
                                #    rollout_fn,
                                   task,
                                   dynamics_model,
                                   sampling_policy,
                                #    value_function,
                                   vf, qf,
                                   hotstart,
                                   state_batch_size,
                                #    num_instances,
                                   squash_fn,
                                   cov_type,
                                   seed,
                                   sample_params=sample_params,
                                   tensor_args=tensor_args)
        self.beta = beta
        self.alpha = alpha  # 0 means control cost is on, 1 means off
        self.update_cov = update_cov
        self.kappa = kappa
        self.visual_traj = visual_traj
        self.normalize_returns = normalize_returns
        self.log_B = torch.log(torch.tensor([self.num_particles], device=self.device)).item()

    def _update_distribution(self, trajectories):
        """
           Update moments in the direction using sampled
           trajectories


        """
        # costs = trajectories["costs"].to(**self.tensor_args)
        # vis_seq = trajectories[self.visual_traj].to(**self.tensor_args)
        actions = trajectories["actions"].to(**self.tensor_args)
        # value_preds = trajectories['value_preds']

        with record_function('mppi:exp_util'):
            w = self._exp_util(trajectories)

        #Update best action
        with record_function('mppi:mean_update'):
            best_idx = torch.argmax(w, dim=1)
            # best_idx = torch.argmax(w, dim=0)
            self.best_idx = best_idx
            self.best_traj = torch.index_select(actions, 1, best_idx)[:,0]#.squeeze(1)
            # self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0) #[:,0]#.squeeze(1)
            # self.best_traj = actions[:, self.best_idx]
            # top_values, top_idx = torch.topk(self.total_costs, 10)
            # self.top_values = top_values
            # self.top_idx = top_idx
            # self.top_trajs = torch.index_select(vis_seq, 1, top_idx).squeeze(1) #.squeeze(0)
            
            # weighted_seq = w.T * actions.T
            # mean_update = torch.sum(weighted_seq.T, dim=1)
            w = w.unsqueeze(-1).unsqueeze(-1)
            weighted_seq = w * actions
            mean_update = torch.sum(weighted_seq, dim=1)

            self.mean_action.data = (1.0 - self.step_size_mean) * self.mean_action.data +\
                self.step_size_mean * mean_update


        with record_function('mppi:cov_update'):
            # assert torch.allclose(weighted_seq.T, weighted_seq_2)
            # new_mean = sum_seq
            #Update Covariance
            if self.update_cov: 
                delta = actions - self.mean_action.unsqueeze(1)
                if self.cov_type == 'sigma_I':
                    #weighted_delta = w * (delta ** 2).T
                    #cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0))
                    #print(cov_update.shape, self.cov_action)
                    raise NotImplementedError('Need to implement covariance update of form sigma*I')
                
                elif self.cov_type == 'diag_AxA':
                    #Diagonal covariance of size AxA
                    # weighted_delta = w * (delta ** 2).T
                    weighted_delta = w * (delta**2)
                    # assert(torch.allclose(weighted_delta, weighted_delta_2))
                    # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0))
                    #sum across batch dimension and mean across temporal dimension aka horizon
                    cov_update = torch.mean(torch.sum(weighted_delta, dim=1), dim=1)
                    # cov_update = torch.mean(torch.sum(weighted_delta, dim=0), dim=0)
                elif self.cov_type == 'diag_HxH':
                    weighted_delta = w * (delta**2)
                    # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0))
                    #sum across batch dimension and mean across temporal dimension aka horizon
                    cov_update = torch.sum(weighted_delta, dim=1)
                    raise NotImplementedError('Need to test/debug covariance update of form diag_HxH')
                elif self.cov_type == 'full_AxA':
                    #Full Covariance of size AxA
                    delta = delta.unsqueeze(-1)
                    cov_update = torch.matmul(delta, delta.transpose(-1,-2))
                    cov_update = w.unsqueeze(-1) * cov_update
                    num_samples = cov_update.shape[1] + cov_update.shape[2]
                    cov_update = torch.sum(torch.sum(cov_update, dim=1), dim=1) / num_samples*1.0
                    # cov_update = torch.sum(torch.sum(cov_update, dim=1), dim=1) /  (cov_update.shape[1] + cov_update.shape[2])
                    # cov_update = torch.mean(torch.sum(cov_update, dim=0), dim=0)
                    ####TO TEST: OR
                    #### cov_update = torch.sum(torch.sum(cov_update, dim=1), dim=1) / (cov_update.shape[1] + cov_update.shape[2])
                    #### cov_update = torch.mean(cov_update.view(cov_update.shape[0]*cov_update.shape[1],-1,-1), dim=0)

                    # weighted_delta = torch.sqrt(w.unsqueeze(-1).unsqueeze(-1)) * delta
                    # weighted_delta = weighted_delta.view(self.num_instances, self.num_particles*self.horizon, -1)
                    # cov_update = torch.matmul(weighted_delta.transpose(-2,-1), weighted_delta) / self.horizon

                    # weighted_delta_2 = torch.sqrt(w.unsqueeze(-1).unsqueeze(-1)) * delta
                    # weighted_delta_2 = weighted_delta_2.unsqueeze(-1)
                    # cov_update_2 = torch.matmul(weighted_delta_2, weighted_delta_2.transpose(-1,-2))
                    # cov_update_2 = torch.mean(torch.sum(cov_update_2, dim=1), dim=1)
                    # assert torch.allclose(cov_update, cov_update_2)

                    # assert torch.allclose(cov_update_2, cov_update_3)

                    # weighted_delta = weighted_delta.T.reshape((self.horizon * self.num_particles, self.d_action))
                    # cov_update = torch.matmul(weighted_delta.T, weighted_delta) / self.horizon
                elif self.cov_type == 'full_HAxHA':# and self.sample_type != 'stomp':
                    weighted_delta = torch.sqrt(w) * delta.view(delta.shape[0], delta.shape[1] * delta.shape[2]).T #.unsqueeze(-1)
                    cov_update = torch.matmul(weighted_delta, weighted_delta.T)
                    raise NotImplementedError('Need to test/debug covariance update of form full_HAxHA')
                    
                    # weighted_cov = w * (torch.matmul(delta_new, delta_new.transpose(-2,-1))).T
                    # weighted_cov = w * cov.T
                    # cov_update = torch.sum(weighted_cov.T,dim=0)
                    #
                #elif self.sample_type == 'stomp':
                #    weighted_delta = w * (delta ** 2).T
                #    cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
                #    self.cov_action = (1.0 - self.step_size_cov) * self.cov_action +\
                #        self.step_size_cov * cov_update
                #    #self.scale_tril = torch.sqrt(self.cov_action)
                #    return
                else:
                    raise ValueError('Unidentified covariance type in update_distribution')
                
                self.cov_action.data = (1.0 - self.step_size_cov) * self.cov_action.data +\
                    self.step_size_cov * cov_update

        # self.mean_action.data = (1.0 - self.step_size_mean) * self.mean_action.data +\
        #     self.step_size_mean * mean_update
                
        return dict(
            mean=self.mean_action, cov=self.full_cov, 
            scale_tril=self.full_scale_tril)

        
    def _shift(self, shift_steps):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        if(shift_steps == 0):
            return
        super()._shift(shift_steps)

        if self.update_cov:
            with record_function('mppi: cov_shift'):
                if self.cov_type == 'sigma_I':
                    self.cov_action += self.kappa #* self.init_cov_action
                    self.scale_tril = torch.sqrt(self.cov_action)
                    # self.inv_cov_action = 1.0 / self.cov_action

                elif self.cov_type == 'diag_AxA':
                    self.cov_action += self.kappa #* self.init_cov_action
                    #self.cov_action[self.cov_action < 0.0005] = 0.0005
                    self.scale_tril = torch.sqrt(self.cov_action)
                    # self.inv_cov_action = 1.0 / self.cov_action
                    
                elif self.cov_type == 'full_AxA':
                    self.cov_action += self.kappa*self.I
                    # self.scale_tril, = matrix_cholesky(self.cov_action) # torch.cholesky(self.cov_action) #
                    self.scale_tril, _ = torch.linalg.cholesky_ex(self.cov_action)
                    # self.scale_tril = torch.cholesky(self.cov_action)
                    # self.inv_cov_action = torch.cholesky_inverse(self.scale_tril)
                
                elif self.cov_type == 'full_HAxHA':
                    self.cov_action += self.kappa * self.I

                    #shift covariance up and to the left
                    # self.cov_action = torch.roll(self.cov_action, shifts=(-self.d_action, -self.d_action), dims=(0,1))
                    # self.cov_action = torch.roll(self.cov_action, shifts=(-self.d_action, -self.d_action), dims=(0,1))
                    # #set bottom A rows and right A columns to zeros
                    # self.cov_action[-self.d_action:,:].zero_()
                    # self.cov_action[:,-self.d_action:].zero_()
                    # #set bottom right AxA block to init_cov value
                    # self.cov_action[-self.d_action:, -self.d_action:] = self.init_cov*self.I2 

                    shift_dim = shift_steps * self.d_action
                    I2 = torch.eye(shift_dim, **self.tensor_args)
                    self.cov_action = torch.roll(self.cov_action, shifts=(-shift_dim, -shift_dim), dims=(0,1))
                    #set bottom A rows and right A columns to zeros
                    self.cov_action[-shift_dim:,:].zero_()
                    self.cov_action[:,-shift_dim:].zero_()
                    #set bottom right AxA block to init_cov value
                    self.cov_action[-shift_dim:, -shift_dim:] = self.init_cov*I2 
                    #update cholesky decomp
                    self.scale_tril = torch.linalg.cholesky(self.cov_action)
                    raise NotImplementedError('Need to test/debug covariance update of form full_HAxHA')

    # def _exp_util(self, costs, actions, value_preds):
    def _exp_util(self, trajectories):
        """
            Calculate weights using exponential utility
        """
        # costs = trajectories['costs'].to(self.device)
        # value_preds = trajectories['value_preds']
        # if value_preds is not None:
        #     costs[..., -1] = value_preds[..., -1] 
        # traj_returns = cost_to_go(costs, self.gammalam_seq)
        # # if not self.time_based_weights: traj_returns = traj_returns[:,0]
        # traj_returns = traj_returns[...,0]
        traj_returns = self._compute_traj_returns(trajectories, normalize=self.normalize_returns)
        #control_costs = self._control_costs(actions)
        # total_returns = traj_returns #+ self.beta * control_costs
        # normalized_traj_returns = traj_returns
        # if self.normalize_returns:
        #     max_return = torch.max(traj_returns, dim=-1)[0][:,None]
        #     min_return = torch.min(traj_returns, dim=-1)[0][:,None]
        #     normalized_traj_returns = (traj_returns - min_return) / (max_return - min_return)
        
        # calculate soft-max for weights
        # returns_weight = normalized_traj_returns if self.normalize_returns else traj_returns
        w = torch.softmax((-1.0/self.beta) * traj_returns, dim=1) 
        # calculate soft optimal value (using non-normalized returns)
        # val = -1.0 * self.beta * (torch.logsumexp((-1.0/self.beta) * traj_returns, dim=1) - self.log_B)
        # val = traj_returns.mean(dim=1)
        # print(self.log_B, torch.log(torch.tensor([traj_returns.shape[1]], device=self.device)))
        # val1 = -1.0 * self.beta * (torch.logsumexp((-1.0/self.beta) * traj_returns, dim=1))
        # val2 = -1.0 * self.beta * scipy.special.logsumexp((-1.0/self.beta) * traj_returns.cpu().numpy(), axis=1, b=(1.0/traj_returns.shape[1]))
        # import pdb; pdb.set_trace()
        self.total_costs = traj_returns
        return w #, val


    def _compute_traj_returns(self, trajectories, normalize:bool=False)->torch.Tensor:
        # import pdb; pdb.set_trace()
        costs = trajectories["costs"]
        value_preds = trajectories['value_preds']
            
        terminals = trajectories['terminals'].float()
        term_cost = trajectories['term_cost']
        # import pdb; pdb.set_trace()
        costs = costs + term_cost
        replace_last_value = False
        # print(value_preds)
        if value_preds is not None:
            if value_preds.shape != costs.shape:
                value_preds = torch.logsumexp(value_preds, dim=0, keepdim=True)
            # value_preds = value_preds * (1. - terminals) + costs * terminals
            if replace_last_value:
                costs[...,-1] = value_preds[...,-1]
            # costs[..., -1] += value_preds[..., -1]
            costs += value_preds #* (1-terminals) #TODO: Check weighting
        traj_returns = cost_to_go(costs, self.gammalam_seq)

        # if not self.time_based_weights: traj_returns = traj_returns[:,0]
        traj_returns = traj_returns[...,0] #- value_preds[...,0]
        if normalize:
            max_return = torch.max(traj_returns, dim=-1)[0][:,None]
            min_return = torch.min(traj_returns, dim=-1)[0][:,None]
            traj_returns = (traj_returns - min_return) / (max_return - min_return + 1e-12)

        return traj_returns
       
    def _calc_val(self, trajectories):
        traj_returns = self._compute_traj_returns(trajectories, normalize=False)
        # if value_preds is not None:
        #     costs[...,-1] = value_preds[...,-1] 

        # traj_returns = cost_to_go(costs, self.gammalam_seq)[..., 0]
        # traj_returns = self.normalize_traj_returns(costs)

        # calculate log-sum-exp
        # c = (-1.0/self.beta) * total_costs.copy()
        # cmax = np.max(c)
        # c -= cmax
        # c = np.exp(c)
        # val1 = cmax + np.log(np.sum(c)) - np.log(c.shape[0])
        # val1 = -self.beta * val1

        # val = -self.beta * scipy.special.logsumexp((-1.0/self.beta) * total_costs, b=(1.0/total_costs.shape[0]))
        # val = -self.beta * (torch.logsumexp((-1.0/self.beta) * traj_returns, dim=1) - self.log_B)
        # val = -1.0 * self.beta * (torch.logsumexp((-1.0/self.beta) * traj_returns, dim=1) - self.log_B)

        val = traj_returns.mean(1)
        return val

    
    # def get_optimal_value(self, state):
    #     trajectories = self.generate_rollouts(state)
    #     return self._calc_val(trajectories)


    def _control_costs(self, actions):
        if self.alpha == 1:
            # if not self.time_based_weights:
            return torch.zeros(actions.shape[0], **self.tensor_args)
        else:
            # u_normalized = self.mean_action.dot(np.linalg.inv(self.cov_action))[np.newaxis,:,:]
            # control_costs = 0.5 * u_normalized * (self.mean_action[np.newaxis,:,:] + 2.0 * delta)
            # control_costs = np.sum(control_costs, axis=-1)
            # control_costs = cost_to_go(control_costs, self.gamma_seq)
            # # if not self.time_based_weights: control_costs = control_costs[:,0]
            # control_costs = control_costs[:,0]
            delta = actions - self.mean_action.unsqueeze(0)
            u_normalized = self.mean_action.matmul(self.full_inv_cov).unsqueeze(0)
            control_costs = 0.5 * u_normalized * (self.mean_action.unsqueeze(0) + 2.0 * delta)
            control_costs = torch.sum(control_costs, dim=-1)
            control_costs = cost_to_go(control_costs, self.gamma_seq)
            control_costs = control_costs[:,0]
        return control_costs




