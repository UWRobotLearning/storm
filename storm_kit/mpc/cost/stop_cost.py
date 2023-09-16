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
import torch.nn as nn
# import torch.nn.functional as F
from .gaussian_projection import GaussianProjection

class StopCost(nn.Module):
    def __init__(self, device=torch.device('cpu'),
                 max_limit=None, max_nlimit=None, weight=1.0,
                 traj_dt=None, **kwargs):
        
        super(StopCost, self).__init__()
        self.device = device
        self.weight = torch.as_tensor(weight, device=self.device)
        self.traj_dt = traj_dt
        
        # compute max velocity across horizon:
        self.horizon = self.traj_dt.shape[0]
        sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), device=self.device)).T

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
