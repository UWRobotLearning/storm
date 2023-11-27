from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch


class NormCost(nn.Module):
    def __init__(
            self,
            weight:float=1.0,
            norm_type: str = 'squared_l2',
            hinge_val: float = 0.0,
            device: torch.device = torch.device('cpu'),
            ):
        super().__init__()
        self.norm_type = norm_type
        self.device = device
        self.weight = weight
        self.hinge_val = hinge_val
        self.log_two = torch.log(torch.tensor([2.0], device=self.device))

    @torch.jit.export
    def forward(self, x:torch.Tensor, hinge_x:Optional[torch.Tensor]=None, keepdim:bool=False) -> torch.Tensor:
        
        if self.norm_type == 'l2':
            dist = torch.norm(x, p=2, dim=-1, keepdim=keepdim)
        elif self.norm_type == 'squared_l2':
            dist = (torch.sum(torch.square(x), dim=-1, keepdim=keepdim))
        elif self.norm_type == 'l1':
            dist = torch.norm(x, p=1, dim=-1, keepdim=keepdim)
        elif self.norm_type == 'logcosh':
            #computes logcosh(x) using softplus and +ve,-ve splitting
            # for numerical stability
            dist = torch.where(x > 0, F.softplus(-2.0 * x) + x - self.log_two, F.softplus(2.0 * x) - x - self.log_two)
            dist = torch.sum(dist, dim=-1, keepdim=keepdim)
        elif self.norm_type == 'smooth_l1':
            l1_dist = torch.norm(x, p=1, dim=-1)
            dist = None
            raise NotImplementedError
        else: raise NotImplementedError

        if hinge_x is not None:
            #if dist is above hinge val we set it to zero
            hinge_mask = hinge_x > self.hinge_val
            dist[hinge_mask] = 0.0

        return self.weight * dist