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
import torch
import numpy as np
import copy
from typing import Tuple, Optional, Dict

class AlphaBetaFilter(object):
    def __init__(self, filter_coeff=0.4):
        self.raw_state = None
        self.filter_coeff = filter_coeff
    
    def filter(self, raw_state):
        self.raw_state = (1 - self.filter_coeff) * self.raw_state + self.filter_coeff * raw_state

    def two_state_filter(self, raw_state1, raw_state2):
        new_state = self.filter_coeff * raw_state1 + (1 - self.filter_coeff) * raw_state2

        return new_state

class RobotStateFilter(object):
    def __init__(
            self, 
            filter_keys=['position', 'velocity','acceleration'], 
            filter_coeff={'position': 0.1, 'velocity':0.1,'acceleration':0.1}, 
            n_dofs=7,
            dt=0.1):
        
        self.prev_filtered_state = None
        self.filtered_state = None
        self.filter_coeff = filter_coeff
        self.filter_keys = filter_keys
        self.dt = dt
        self.n_dofs = n_dofs

    def filter_state(self, raw_state, dt=None):
        dt = self.dt if dt is None else dt
        if self.filtered_state is None:
            self.filtered_state = copy.deepcopy(raw_state)
            if 'acceleration' in self.filter_keys:
                self.filtered_state['acceleration'] = 0.0* raw_state['position']
            #return self.filtered_state
        self.prev_filtered_state = copy.deepcopy(self.filtered_state)
        for k in self.filter_keys:
            if(k in raw_state.keys()):
                self.filtered_state[k] = self.filter_coeff[k] * raw_state[k] + (1.0 - self.filter_coeff[k]) * self.filtered_state[k]
        if 'acceleration' in self.filter_keys:# and 'acceleration' not in raw_state):
            self.filtered_state['acceleration'] = (self.filtered_state['velocity'] - self.prev_filtered_state['velocity']) / dt
        return self.filtered_state

# @torch.jit.script     
class JointStateFilter(object):
    
    def __init__(
            self, 
            # raw_joint_state: Optional[torch.Tensor] = None, 
            # filter_keys: Tuple[str] = ['position','velocity','acceleration'],
            filter_coeff: Dict[str, float], 
            n_dofs: int,
            dt: float = 0.1, 
            # bounds: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = torch.device('cpu')
            ):
        
        self.device = device
        self.n_dofs = n_dofs
        # self.internal_jnt_state = torch.empty((self.n_dofs), device=self.device)
        # if raw_joint_state is not None:
        #     self.internal_jnt_state = raw_joint_state.clone().to(self.device)
        self.filter_coeff = filter_coeff 
        self.filter_keys = filter_coeff.keys()
        self.internal_jnt_state: Dict[str, torch.Tensor] = {} 
        for k in self.filter_keys:
            self.internal_jnt_state[k] = torch.empty((), device=self.device) 
        
        # if isinstance(filter_coeff, float):
        #     for k in filter_keys:
        #         self.filter_coeff[k] = filter_coeff
        # else:
        #     self.filter_coeff = filter_coeff
        self.dt = dt
        self.prev_cmd_qdd = None
        self.initial_step = True 
    
    def filter_joint_state(self, raw_joint_state: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:

        if self.initial_step:
            for k in raw_joint_state.keys():
                self.internal_jnt_state[k] = raw_joint_state[k].clone().to(self.device)
            self.initial_step = False
            return self.internal_jnt_state
        # q_pos_raw = raw_joint_state[..., 0:self.n_dofs]
        # q_vel_raw = raw_joint_state[..., self.n_dofs:2*self.n_dofs]
        # q_acc_raw = raw_joint_state[..., 2*self.n_dofs:3*self.n_dofs]

        # q_pos_internal = self.internal_jnt_state[..., 0:self.n_dofs]
        # q_vel_internal = self.internal_jnt_state[..., self.n_dofs:2*self.n_dofs]
        # q_acc_internal = self.internal_jnt_state[..., 2*self.n_dofs:3*self.n_dofs]

        # # if 'position' in self.filter_keys:
        # coeff = self.filter_coeff['position']
        # self.internal_jnt_state[..., 0:self.n_dofs] = coeff * q_pos_raw + (1.0 - coeff) * q_pos_internal

        # # if 'velocity' in self.filter_keys:
        # coeff = self.filter_coeff['velocity']
        # self.internal_jnt_state[..., self.n_dofs:2*self.n_dofs] = coeff * q_vel_raw + (1.0 - coeff) * q_vel_internal

        # # if 'acceleration' in self.filter_keys:
        # coeff = self.filter_coeff['acceleration']
        # self.internal_jnt_state[..., 2*self.n_dofs:3*self.n_dofs] = coeff * q_acc_raw + (1.0 - coeff) * q_acc_internal

        # for k in self.filter_keys:
        for k in raw_joint_state.keys():
            if k in self.filter_keys:
                self.internal_jnt_state[k] = self.filter_coeff[k] * raw_joint_state[k] + (1.0 - self.filter_coeff[k]) * self.internal_jnt_state[k]
            else:
                self.internal_jnt_state[k] = raw_joint_state[k].clone()

        return self.internal_jnt_state

    # def forward_predict_internal_state(self, dt: Optional[float] = None):
    #     if self.prev_cmd_qdd is None:
    #         return
    #     dt = self.dt if dt is None else dt 
    #     self.internal_jnt_state[...,2*self.n_dofs:3*self.n_dofs] = self.prev_cmd_qdd
    #     self.internal_jnt_state[...,self.n_dofs:2*self.n_dofs] += self.prev_cmd_qdd * dt
    #     self.internal_jnt_state[...,0:self.n_dofs] += self.internal_jnt_state[...,self.n_dofs:2*self.n_dofs] * dt
        # self.internal_jnt_state['acceleration'] = self.prev_cmd_qdd
        # self.internal_jnt_state['velocity'] = self.internal_jnt_state['velocity'] + self.prev_cmd_qdd * dt
        # self.internal_jnt_state['position'] = self.internal_jnt_state['position'] + self.internal_jnt_state['velocity'] * dt
        

    def predict_internal_state(
            self, 
            qdd_des: Optional[torch.Tensor] = None, 
            dt: Optional[float] = None) -> Dict[str, torch.Tensor]:

        if qdd_des is None:
            return self.internal_jnt_state
        dt = self.dt if dt is None else dt 
        self.internal_jnt_state['q_acc'] = qdd_des
        self.internal_jnt_state['q_vel'] += self.internal_jnt_state['q_acc'] * dt        
        self.internal_jnt_state['q_pos'] += self.internal_jnt_state['q_vel'] * dt

        return self.internal_jnt_state
        # self.internal_jnt_state[...,2*self.n_dofs:3*self.n_dofs] = qdd_des
        # self.internal_jnt_state[...,self.n_dofs:2*self.n_dofs] += self.internal_jnt_state[...,2*self.n_dofs:3*self.n_dofs] * dt
        # self.internal_jnt_state[...,0:self.n_dofs] += self.internal_jnt_state[...,self.n_dofs:2*self.n_dofs] * dt

    # def integrate_jerk(self, qddd_des, raw_joint_state, dt=None):
    #     dt = self.dt if dt is None else dt 
    #     self.filter_joint_state(raw_joint_state)
    #     self.internal_jnt_state['acceleration'] = self.internal_jnt_state['acceleration'] + qddd_des * dt
    #     self.internal_jnt_state['velocity'] = self.internal_jnt_state['velocity'] + self.internal_jnt_state['acceleration'] * dt
    #     self.internal_jnt_state['position'] = self.internal_jnt_state['position'] + self.internal_jnt_state['velocity'] * dt
    #     self.prev_cmd_qdd = self.internal_jnt_state['acceleration']
    #     return self.internal_jnt_state

    # def integrate_acc(self, qdd_des, raw_joint_state=None, dt=None):
    #     dt = self.dt if dt is None else dt
    #     if raw_joint_state is not None:
    #         self.filter_joint_state(raw_joint_state)
    #     self.internal_jnt_state['acceleration'] = qdd_des
    #     self.internal_jnt_state['velocity'] = self.internal_jnt_state['velocity'] + qdd_des * dt
    #     self.internal_jnt_state['position'] = self.internal_jnt_state['position'] + self.internal_jnt_state['velocity'] * dt
    #     self.prev_cmd_qdd = self.internal_jnt_state['acceleration']
    #     return self.internal_jnt_state

    # def integrate_vel(self, qd_des, raw_joint_state, dt=None):
    #     dt = self.dt if dt is None else dt
    #     self.filter_joint_state(raw_joint_state)
    #     self.internal_jnt_state['velocity'] = qd_des #self.internal_jnt_state['velocity'] + qdd_des * dt
    #     self.internal_jnt_state['position'] = self.internal_jnt_state['position'] + self.internal_jnt_state['velocity'] * dt

    #     return self.internal_jnt_state

    # def integrate_pos(self, q_des, raw_joint_state, dt=None):
    #     dt = self.dt if dt is None else dt
    #     self.filter_joint_state(raw_joint_state)

    #     self.internal_jnt_state['velocity'] = (q_des - self.internal_jnt_state['position']) / dt
    #     self.internal_jnt_state['position'] = self.internal_jnt_state['position'] + self.internal_jnt_state['velocity'] * dt

    #     # This needs to also update the acceleration via finite differencing.
    #     raise NotImplementedError

    #     return self.internal_jnt_state

    def reset(self):
        self.internal_jnt_state = {}
        self.initial_step = True
