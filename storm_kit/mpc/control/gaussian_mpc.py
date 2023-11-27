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
"""
MPC with open-loop Gaussian policies
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from .control_base import Controller
from .control_utils import generate_noise, scale_ctrl, gaussian_entropy, matrix_cholesky
from .sample_libs import StompSampleLib, HaltonSampleLib, RandomSampleLib, HaltonStompSampleLib, MultipleSampleLib

class GaussianMPC(Controller):
    """
        .. inheritance-diagram:: OLGaussianMPC
           :parts: 1
    """    
    def __init__(self, 
                 d_action,                
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
                 null_act_frac=0.,
                 rollout_fn=None,
                 hotstart=True,
                 num_instances=1,
                 squash_fn='clamp',
                 cov_type='sigma_I',
                 seed=0,
                 sample_params={'type': 'halton', 'fixed_samples': True, 'seed':0, 'filter_coeffs':None},
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        """
        Parameters
        __________
        base_action : str
            Action to append at the end when shifting solution to next timestep
            'random' : appends random action
            'null' : appends zero action
            'repeat' : repeats second to last action
        num_particles : int
            Number of action sequences sampled at every iteration
        """

        super(GaussianMPC, self).__init__(
            d_action,
            action_lows,
            action_highs,
            horizon,
            gamma,
            td_lam,
            n_iters,
            rollout_fn,
            hotstart,
            num_instances,
            seed,
            tensor_args)
    
        self.init_cov = init_cov 
        self.init_mean = init_mean.clone().to(**self.tensor_args)
        if self.init_mean.ndim == 2:
            self.init_mean = self.init_mean.unsqueeze(0).repeat(self.num_instances, 1, 1)
        self.cov_type = cov_type
        self.base_action = base_action
        self.num_particles = num_particles
        self.step_size_mean = step_size_mean
        self.step_size_cov = step_size_cov
        self.squash_fn = squash_fn

        self.null_act_frac = null_act_frac
        self.num_null_particles = round(int(null_act_frac * self.num_particles * 1.0))
        # self.num_neg_particles = round(int(null_act_frac * self.num_particles)) - self.num_null_particles
        self.num_nonzero_particles = self.num_particles - self.num_null_particles# - self.num_neg_particles

        self.sample_params = sample_params
        self.sample_type = sample_params['type']
        # initialize sampling library:
        if sample_params['type'] == 'stomp':
            self.sample_lib = StompSampleLib(self.horizon, self.d_action, tensor_args=self.tensor_args)
            self.sample_shape = torch.Size([self.num_nonzero_particles - 2])
            self.i_ha = torch.eye(self.d_action, **self.tensor_args).repeat(1, self.horizon)

        elif sample_params['type'] == 'halton':
            self.sample_lib = HaltonSampleLib(
                self.num_instances, self.horizon, 
                self.d_action, device=self.tensor_args['device'],
                **self.sample_params)
            self.sample_shape = torch.Size([self.num_nonzero_particles - 2])

        elif sample_params['type'] == 'random':
            self.sample_lib = RandomSampleLib(
                self.num_instances, self.horizon, 
                self.d_action, device=self.tensor_args['device'],
                **self.sample_params)
            self.sample_shape = torch.Size([self.num_nonzero_particles - 2])

        elif sample_params['type'] == 'multiple':
            self.sample_lib = MultipleSampleLib(
                self.num_instances, self.horizon, 
                self.d_action, device=self.tensor_args['device'], 
                **self.sample_params)
            self.sample_shape = torch.Size([self.num_nonzero_particles - 2])

        self.stomp_matrix = None #self.sample_lib.stomp_cov_matrix
        # initialize covariance types:
        if self.cov_type == 'full_HAxHA':
            self.I = torch.eye(self.horizon * self.d_action, **self.tensor_args).unsqueeze(0).repeat(self.num_instances, 1)
        else: # AxA
            self.I = torch.eye(self.d_action, **self.tensor_args).unsqueeze(0).repeat(self.num_instances, 1, 1)
        
        self.Z_seq = torch.zeros(self.num_instances, 1, self.horizon, self.d_action, **self.tensor_args)

        self.reset_distribution()
        
        if self.num_null_particles > 0:
            self.null_act_seqs = torch.zeros(self.num_instances, self.num_null_particles, self.horizon, self.d_action, **self.tensor_args)
            
        self.delta = None

    # def _get_action_seq(self, deterministic:bool=True):
    #     if deterministic:
    #         act_seq = self.mean_action.data#.clone()
    #     else:
    #         print('in here!!!!!')
    #         delta = self.generate_noise(shape=torch.Size(self.horizon),
    #                                     base_seed=self.seed_val + 123 * self.num_steps)

    #         act_seq = self.mean_action.data + torch.matmul(delta, self.full_scale_tril)
    #         print(delta.shape, act_seq.shape, self.mean_action.data.shape, self.full_scale_tril.shape)
    #     act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

    #     return act_seq

    def sample(self, state, calc_val:bool=False, shift_steps:int=1, n_iters=None, deterministic:bool=False, num_samples:int=1):
        distrib_info, value, aux_info =  self.optimize(state, calc_val, shift_steps, n_iters)
        
        if deterministic:
            return distrib_info['mean'].data.unsqueeze(0) , value, aux_info
        
        samples = self.generate_noise(distrib_info, num_samples)
        return samples, value, aux_info

    def generate_noise(self, distrib_info, num_samples):
        """
            Generate correlated noisy samples using autoregressive process
        """
        mean = distrib_info['mean']
        scale_tril = distrib_info['scale_tril']
        random_samples = torch.randn(num_samples, mean.shape[0], mean.shape[1], mean.shape[2], device=self.device)
        scaled_samples = torch.matmul(random_samples, scale_tril)
        return mean + scaled_samples

        
    def sample_actions(self, state=None):
        delta = self.sample_lib.get_samples(sample_shape=self.sample_shape, base_seed=self.seed_val + self.num_steps)
        #add zero-noise seq so mean is always a part of samples
        delta = torch.cat((delta, self.Z_seq), dim=1)
        #TODO: Is this right?
        # delta = delta.unsqueeze(0).repeat(self.num_instances, 1, 1, 1)
        # samples could be from HAxHA or AxA:
        # We reshape them based on covariance type:
        # if cov is AxA, then we don't reshape samples as samples are: N x H x A
        # if cov is HAxHA, then we reshape
        if self.cov_type == 'full_HAxHA':
            # delta: N * H * A -> N * HA
            delta = delta.view(delta.shape[0], delta.shape[1], self.horizon * self.d_action)
        
        scaled_delta = torch.matmul(delta, self.full_scale_tril.unsqueeze(1)).view(delta.shape[0],
                                                                      delta.shape[1],
                                                                      self.horizon,
                                                                      self.d_action)
        # debug_act = delta[:,:,:,0].cpu().numpy()

        act_seq = self.mean_action.unsqueeze(1) + scaled_delta
        # act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)
        append_acts = self.best_traj.unsqueeze(1)

        #append zero actions (for stopping)
        if self.num_null_particles > 0:
            # zero particles:

            # negative action particles:
            # neg_action = -1.0 * self.mean_action.unsqueeze(1)
            # print(neg_action.shape)
            # neg_act_seqs = neg_action.expand(1, self.num_neg_particles,-1,-1)
            # print(append_acts.shape, self.null_act_seqs.shape, neg_act_seqs.shape)

            # append_acts = torch.cat((append_acts, self.null_act_seqs, neg_act_seqs),dim=1)
            append_acts = torch.cat((append_acts, self.null_act_seqs),dim=1)

        act_seq = torch.cat((act_seq, append_acts), dim=1)
        return act_seq

    def generate_rollouts(self, state):
        """
            Samples a batch of actions, rolls out trajectories for each particle
            and returns the resulting observations, costs,  
            actions

            Parameters
            ----------
            state : dict or np.ndarray
                Initial state to set the simulation env to
         """
        
        act_seq = self.sample_actions(state=state) # sample noise from covariance of current control distribution
        trajectories = self._rollout_fn(state, act_seq)
        return trajectories
    
    def _shift(self, shift_steps=1):
        """
            Predict mean for the next time step by
            shifting the current mean forward by one step
        """
        if shift_steps == 0:
            return
        # self.new_mean_action = self.mean_action.clone()
        # self.new_mean_action[:-1] = #self.mean_action[1:]
        self.mean_action.data = self.mean_action.data.roll(-shift_steps, 1)
        self.best_traj = self.best_traj.roll(-shift_steps, 1)
        
        if self.base_action == 'random':
            self.mean_action.data[:, -1, :] = self.generate_noise(shape=torch.Size((1, 1)), 
                                                       base_seed=self.seed_val + 123*self.num_steps)
            self.best_traj[:, -1, :] = self.generate_noise(shape=torch.Size((1, 1)), 
                                                     base_seed=self.seed_val + 123*self.num_steps)
        elif self.base_action == 'null':
            self.mean_action.data[:, -shift_steps:].zero_() 
            self.best_traj[:, -shift_steps:].zero_()
        elif self.base_action == 'repeat':
            self.mean_action.data[:, -shift_steps:] = self.mean_action.data[:, -shift_steps -1].unsqueeze(1).clone()
            self.best_traj[:, -shift_steps:] = self.best_traj[:, -shift_steps -1 ].unsqueeze(1).clone()
            #self.mean_action[-1] = self.mean_action[-2].clone()
            #self.best_traj[-1] = self.best_traj[-2].clone()
        else:
            raise NotImplementedError("invalid option for base action during shift")
        # self.mean_action = self.new_mean_action

    def reset_mean(self):
        # self.mean_action = self.init_mean.clone()
        # self.best_traj = self.mean_action.clone()
        self.mean_action = nn.Parameter(data=self.init_mean.clone(), requires_grad=False)
        self.best_traj = self.mean_action.data.clone()

    def reset_covariance(self):

        if self.cov_type == 'sigma_I':
            init_cov = torch.tensor(self.init_cov, **self.tensor_args).unsqueeze(0).repeat(self.num_instances, 1)
            self.init_cov_action = self.init_cov
            self.cov_action = nn.Parameter(init_cov, requires_grad=False)
            self.inv_cov_action = 1.0 / self.init_cov  
            self.scale_tril = torch.sqrt(self.cov_action)
        
        elif self.cov_type == 'diag_AxA':
            init_cov = torch.tensor([self.init_cov]*self.d_action, **self.tensor_args)
            init_cov = init_cov.unsqueeze(0).repeat(self.num_instances, 1)
            self.init_cov_action = init_cov
            self.cov_action = nn.Parameter(data=self.init_cov_action, requires_grad=False)
            self.inv_cov_action = 1.0 / self.cov_action.data
            self.scale_tril = torch.sqrt(self.cov_action.data)

        
        elif self.cov_type == 'full_AxA':
            self.init_cov_action = torch.diag(torch.tensor([self.init_cov]*self.d_action, **self.tensor_args))
            self.init_cov_action = self.init_cov_action.unsqueeze(0).repeat(self.num_instances, 1, 1)
            self.cov_action = nn.Parameter(data=self.init_cov_action, requires_grad=False)
            self.scale_tril = matrix_cholesky(self.cov_action.data) #torch.cholesky(self.cov_action)
            self.inv_cov_action = torch.cholesky_inverse(self.scale_tril)

        elif self.cov_type == 'full_HAxHA':
            self.init_cov_action = torch.diag(torch.tensor([self.init_cov] * (self.horizon * self.d_action), **self.tensor_args))
            self.init_cov_action = self.init_cov_action.unsqueeze(0).repeat(self.num_instances, 1, 1)
            self.cov_action = nn.Parameter(self.init_cov_action, requires_grad=False)
            self.scale_tril = torch.cholesky(self.cov_action.data)
            self.inv_cov_action = torch.cholesky_inverse(self.scale_tril)
            
        else:
            raise ValueError('Unidentified covariance type in update_distribution')

    def reset_distribution(self):
        """
            Reset control distribution
        """
        self.reset_mean()
        self.reset_covariance()

    def _calc_val(self, cost_seq, act_seq):
        raise NotImplementedError("_calc_val not implemented")


    @property
    def squashed_mean(self):
        return scale_ctrl(self.mean_action, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

    @property
    def full_cov(self):
        if self.cov_type == 'sigma_I':
            return self.cov_action * self.I
        elif self.cov_type == 'diag_AxA':
            return torch.diag_embed(self.cov_action)
        elif self.cov_type == 'full_AxA':
            return self.cov_action
        elif self.cov_type == 'full_HAxHA':
            return self.cov_action
    
    @property
    def full_inv_cov(self):
        if self.cov_type == 'sigma_I':
            return self.inv_cov_action * self.I
        elif self.cov_type == 'diag_AxA':
            return torch.diag_embed(self.inv_cov_action)
        elif self.cov_type == 'full_AxA':
            return self.inv_cov_action
        elif self.cov_type == 'full_HAxHA':
            return self.inv_cov_action

            

    @property
    def full_scale_tril(self):
        if self.cov_type == 'sigma_I':
            return self.scale_tril * self.I
        elif self.cov_type == 'diag_AxA':
            return torch.diag_embed(self.scale_tril)
        elif self.cov_type == 'full_AxA':
            return self.scale_tril
        elif self.cov_type == 'full_HAxHA':
            return self.scale_tril
            

    @property
    def entropy(self):
        # ent_cov = gaussian_entropy(cov=self.full_cov)
        ent_L = gaussian_entropy(L=self.full_scale_tril)
        return ent_L
