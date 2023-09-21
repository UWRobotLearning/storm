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
# DEALINGS IN THE SOFTWARE.
import torch
import torch.nn as nn
from ..model.integration_utils import build_fd_matrix

class FiniteDifferenceCost(nn.Module):
    def __init__(self, device:torch.device('cpu'), weight=1.0, order=1, **kwargs):
        super(FiniteDifferenceCost, self).__init__()
        self.device = device
        self.order = order
        for _ in range(order):
            weight *= weight
        self.weight = weight
        self.device = device
        # build FD matrix
        self.fd_mat = None
        self.t_mat = None


    def forward(self, ctrl_seq, dt):
        """
        ctrl_seq: [B X H X d_act]
        """
        dt[dt == 0.0] = 0.0 #dt[-1]
        dt = 1 / dt
        
        #dt = dt / torch.max(dt)
        dt = torch.abs(dt)
        
        #print(dt)
        dt[dt == float("Inf")] = 0

        dt[dt > 10] = 10
        #dt = dt / torch.max(dt)
        
        dt[dt != dt] = 0.0
        #for _ in range(self.order-1):
        #    dt = dt * dt
        #print(dt)
        inp_device = ctrl_seq.device
        ctrl_seq = ctrl_seq.to(device=self.device)
        
        _, H, _ = ctrl_seq.shape
        H = H - self.order
        dt = dt[:H]
        #
        if self.fd_mat is None or self.fd_mat.shape[0] != H:
            self.fd_mat = build_fd_matrix(H,device=self.device, order=self.order, PREV_STATE=True)
            
        diff = torch.matmul(self.fd_mat,ctrl_seq)
        res = torch.abs(diff)
        cost = res[:,:,-1]
        cost[cost < 0.0001] = 0.0
        cost = self.weight * cost 
        
        return cost





    # def forward(self, ctrl_seq, dt):
    #     """
    #     ctrl_seq: [B X H X d_act]
    #     """
    #     dt[dt == 0.0] = 0.0 #dt[-1]
    #     dt = 1 / dt
        
    #     #dt = dt / torch.max(dt)
    #     dt = torch.abs(dt)
        
    #     #print(dt)
    #     dt[dt == float("Inf")] = 0

    #     dt[dt > 10] = 10
    #     #dt = dt / torch.max(dt)
        
    #     dt[dt != dt] = 0.0
    #     #for _ in range(self.order-1):
    #     #    dt = dt * dt
    #     #print(dt)
    #     inp_device = ctrl_seq.device
    #     ctrl_seq = ctrl_seq.to(device=self.device)
        
    #     _, H, _ = ctrl_seq.shape
    #     H = H - self.order
    #     dt = dt[:H]
    #     #
    #     if self.fd_mat is None or self.fd_mat.shape[0] != H:
    #         self.fd_mat = build_fd_matrix(H,device=self.device, order=self.order, PREV_STATE=True)
            
    #     diff = torch.matmul(self.fd_mat,ctrl_seq)
    #     res = torch.abs(diff)
    #     cost = res[:,:,-1]
    #     cost[cost < 0.0001] = 0.0
    #     cost = self.weight * cost 
        
    #     return cost
