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
from typing import Optional, Dict
import torch
import torch.nn as nn

class StopCost(nn.Module):
    def __init__(self, horizon:int,  dt_traj_params:Dict[str, float],
                 max_limit:Optional[float]=None, max_nlimit:Optional[float]=None, 
                 weight:float=1.0, device=torch.device('cpu'), **kwargs):
        
        super(StopCost, self).__init__()
        self.device = device
        self.weight = torch.as_tensor(weight, device=self.device)
        
        # compute max velocity across horizon:
        self.horizon = horizon #self.traj_dt.shape[0]
        sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), device=self.device)).T


        self.dt_traj_params = dt_traj_params

        # if dt_traj_params is None or self.num_traj_points <= 1:
        #     dt_array = [self.dt] * int(1.0 * self.num_traj_points) 
        if self.horizon <= 1:
            dt_array = [dt_traj_params['base_dt']] * int(1.0 * self.horizon)
        else:
            dt_array = [dt_traj_params['base_dt']] * int(dt_traj_params['base_ratio'] * self.horizon)
            smooth_blending = torch.linspace(dt_traj_params['base_dt'], dt_traj_params['max_dt'], steps=int((1 - dt_traj_params['base_ratio']) * self.horizon)).tolist()
            dt_array += smooth_blending
            self.dt = dt_traj_params['base_dt']

        self.traj_dt = torch.tensor(dt_array, device=self.device)

        if max_nlimit is not None:
            # every timestep max acceleration:
            sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), device=self.device)).T
            delta_vel = self.traj_dt * max_nlimit
            self.max_vel = ((sum_matrix @ delta_vel).unsqueeze(-1))
        
        elif max_limit is not None:
            sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), device=self.device)).T
            delta_vel = torch.ones_like(self.traj_dt) * max_limit
            self.max_vel = ((sum_matrix @ delta_vel).unsqueeze(-1))
        
    def forward(self, vels:torch.Tensor):
        inp_device = vels.device

        vel_abs = torch.abs(vels.to(device=self.device))
        # max velocity threshold:
        vel_abs = vel_abs - self.max_vel
        vel_abs[vel_abs < 0.0] = 0.0
        
        # cost = self.weight * self.proj_gaussian(((torch.sum(torch.square(vel_abs), dim=-1))))
        cost = self.weight * ((torch.sum(torch.square(vel_abs), dim=-1)))

        return cost.to(inp_device)
