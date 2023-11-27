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
from torch.profiler import record_function

eps = 0.01


class ManipulabilityCost(nn.Module):
    def __init__(
            self, 
            # ndofs: int, 
            weight = None, 
            device = torch.device('cpu'), 
            float_dtype = torch.float32, 
            thresh:float = 0.1):
        
        super(ManipulabilityCost, self).__init__() 
        self.device = device
        self.float_dtype = float_dtype
        self.weight = torch.as_tensor(weight, device=device, dtype=float_dtype)
        # self.ndofs = ndofs
        self.thresh = thresh
        self.info = {}
        # self.i_mat = torch.ones((6,1), device=self.device, dtype=self.float_dtype)
    
    def forward(self, jac_batch:torch.Tensor) -> torch.Tensor:
        inp_device = jac_batch.device

        with torch.cuda.amp.autocast(enabled=False):
            J_J_t = torch.matmul(jac_batch, jac_batch.transpose(-2,-1))
            score = torch.sqrt(torch.linalg.det(J_J_t))
            # score = torch.linalg.det(J_J_t)
            # with record_function('manip_cost:chol'):
            #     chol = torch.linalg.cholesky(J_J_t)
            # with record_function('manip_cost:diag'):
            #     chol_diag = torch.diagonal(chol, dim1=-2, dim2=-1)
            # with record_function('manip_cost:prod'):
            #     score = torch.prod(chol_diag, dim=-1)

        score[score != score] = 0.0
        self.info['manip_score'] = score
        score[score > self.thresh] = self.thresh
        score = (self.thresh - score) / self.thresh
        cost = self.weight * score 
        
        return cost.to(inp_device), self.info
    
