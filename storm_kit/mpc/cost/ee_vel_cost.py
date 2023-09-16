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

class EEVelCost(nn.Module):
    def __init__(self, ndofs, device, weight=1.0, vec_weight=[]):
        super(EEVelCost, self).__init__()
        self.ndofs = ndofs
        self.device = device
        self.vel_idxs = torch.arange(self.ndofs,2*self.ndofs, dtype=torch.long, device=self.device)
        self.I = torch.eye(ndofs, device=device)
        self.vec_weight = torch.as_tensor(vec_weight, device=device)
        self.weight = weight
        
    
    def forward(self, state_batch, jac_batch):
        
        inp_device = state_batch.device
        jac_batch = jac_batch.to(self.device)
        
        #use jacobian to get desired delta_q
        J = jac_batch
        qdot = state_batch[:,:,self.ndofs:2 * self.ndofs]

        xdot_current = torch.matmul(J, qdot.unsqueeze(-1)).squeeze(-1)
        
        error = torch.sum(torch.square(self.vec_weight * xdot_current), dim=-1)

        cost = self.weight * self.gaussian_projection(error)
        return cost.to(inp_device)

