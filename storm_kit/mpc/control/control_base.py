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
from abc import abstractmethod
import torch
import torch.nn as nn
from torch.profiler import record_function

class Controller(nn.Module):
    """Base class for sampling based controllers."""

    def __init__(self,
                 d_action,
                 action_lows,
                 action_highs,
                 horizon,
                 gamma:float,
                 td_lam:float,
                 n_iters:int,
                #  rollout_fn=None,
                 hotstart:bool=True,
                 state_batch_size:int=1,
                 seed:int=0,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        """
        Defines an abstract base class for 
        sampling based MPC algorithms.

        Implements the optimize method that is called to 
        generate an action sequence for a given state and
        is common across sampling based controllers

        Attributes:
        
        d_action : int
            size of action space
        action_lows : torch.Tensor 
            lower limits for each action dim
        action_highs : torch.Tensor  
            upper limits for each action dim
        horizon : int  
            horizon of rollouts
        gamma : float
            discount factor
        n_iters : int  
            number of optimization iterations per MPC call
        rollout_fn : function handle  
            rollout policy (or actions) in simulator
            and return states and costs for updating MPC
            distribution
        sample_mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
            'sample' samples from the distribution
        hotstart : bool
            If true, the solution from previous step
            is used to warm start current step
        seed : int  
            seed value
        device: torch.device
            controller can run on both cpu and gpu
        float_dtype: torch.dtype
            floating point precision for calculations
        """
        super().__init__()
        self.tensor_args = tensor_args
        self.device = tensor_args['device']
        self.d_action = d_action
        self.action_lows = torch.as_tensor(action_lows).to(**self.tensor_args)
        self.action_highs = torch.as_tensor(action_highs).to(**self.tensor_args)
        self.horizon = horizon
        self.gamma = gamma
        self.td_lam = td_lam
        self.n_iters = n_iters
        self.gamma_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma] * (horizon - 1)), dim=0).reshape(1, horizon).to(**self.tensor_args)
        self.gammalam_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma*self.td_lam] * (horizon - 1)), dim=0).reshape(1, self.horizon).to(**self.tensor_args)
        # self._rollout_fn = rollout_fn
        self.num_steps = 0
        self.hotstart = hotstart
        self.state_batch_size = state_batch_size
        self.seed_val = seed
        self.trajectories = None
        
    # @abstractmethod
    # def _get_action_seq(self, deterministic:bool=True):
    #     """
    #     Get action sequence to execute on the system based
    #     on current control distribution
        
    #     Args:
    #         mode : {'mean', 'sample'}  
    #             how to choose action to be executed
    #             'mean' plays mean action and  
    #             'sample' samples from the distribution
    #     """        
    #     pass

    def sample_actions(self):
        """
        Sample actions from current control distribution
        """
        raise NotImplementedError('sample_actions funtion not implemented')
    
    @abstractmethod
    def _update_distribution(self, trajectories):
        """
        Update current control distribution using 
        rollout trajectories
        
        Args:
            trajectories : dict
                Rollout trajectories. Contains the following fields
                observations : torch.tensor
                    observations along rollouts
                actions : torch.tensor 
                    actions sampled from control distribution along rollouts
                costs : torch.tensor 
                    step costs along rollouts
        """
        pass

    @abstractmethod
    def _shift(self):
        """
        Shift the current control distribution
        to hotstart the next timestep
        """
        pass

    @abstractmethod
    def reset_distribution(self):
        pass

    def reset(self):
        """
        Reset the controller
        """
        self.num_steps = 0
        self.reset_distribution()

    @abstractmethod
    def _calc_val(self, cost_seq, act_seq):
        """
        Calculate value of state given 
        rollouts from a policy
        """
        pass

    def check_convergence(self):
        """
        Checks if controller has converged
        Returns False by default
        """
        return False
        
    # @property
    # def rollout_fn(self):
    #     return self._rollout_fn
    
    # @rollout_fn.setter
    # def rollout_fn(self, fn):
    #     """
    #     Set the rollout function from 
    #     input function pointer
    #     """
    #     self._rollout_fn = fn
    
    @abstractmethod
    def generate_rollouts(self, state):
        pass
    
    @abstractmethod
    def sample(self, state, shift_steps:int=1, n_iters=None, deterministic:bool=False): #calc_val:bool=False , num_samples:int=1
        pass

    def forward(self, state, shift_steps:int=1, n_iters=None): #calc_val:bool=False
        distrib_info, value, aux_info =  self.optimize(state, shift_steps, n_iters) #calc_val
        return distrib_info, value, aux_info

    def optimize(self, state, shift_steps:int=1, n_iters:int=None): #calc_val:bool=False
        """
        Optimize for best action at current state

        Parameters
        ----------
        state : torch.Tensor
            state to calculate optimal action from
        
        calc_val : bool
            If true, calculate the optimal value estimate
            of the state along with action
                
        Returns
        -------
        action : torch.Tensor
            next action to execute
        value: float
            optimal value estimate (default: 0.)
        info: dict
            dictionary with side-information
        """

        n_iters = n_iters if n_iters is not None else self.n_iters

        aux_info = {} #TODO: Return rollout trajectories? 

        # shift distribution to hotstart from previous timestep
        if self.hotstart:
            with record_function('mpc:hotstart'):
                self._shift(shift_steps)
        else:
            self.reset_distribution()
            

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                for _ in range(n_iters):
                    # generate random simulated trajectories
                    with record_function('mpc:generate_rollouts'):
                        trajectory = self.generate_rollouts(state)

                    # update distribution parameters
                    with record_function("mpc:update_distribution"):
                        distrib_info = self._update_distribution(trajectory)

                    # check if converged
                    if self.check_convergence():
                        break

        self.trajectories = trajectory

        # #calculate optimal value estimate if required
        # value = 0.0
        # if calc_val:
        #     value = self.compute_optimal_value(state, trajectory)
        self.num_steps += 1

        return distrib_info, aux_info #value
    
    @abstractmethod
    def compute_optimal_value(self, state, trajectories=None):
        """
        Calculate optimal value of a state, i.e 
        value under optimal policy. 

        Parameters
        ----------
        state : torch.Tensor
            state to calculate optimal value estimate for
        Returns
        -------
        value : float
            optimal value estimate of the state
        """
        pass    




