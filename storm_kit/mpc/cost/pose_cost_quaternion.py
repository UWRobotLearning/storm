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
from typing import List
from storm_kit.differentiable_robot_model.coordinate_transform import matrix_to_quaternion

class PoseCostQuaternion(nn.Module):
    """ Pose cost using quaternion distance for orienatation 

    .. math::
     
    r  &=  \sum_{i=0}^{num_rows} (R^{i,:} - R_{g}^{i,:})^2 \\
    cost &= \sum w \dot r

    
    """
    weight: List[float]
    hinge_val: float
    convergence_val: List[float]
    def __init__(self, 
                 weight: List[float], 
                 hinge_val: float = 100.0,
                 convergence_val: List[float] = [0.0, 0.0],
                 device:torch.device = torch.device("cpu"),
                 quat_inputs: bool =False):

        super(PoseCostQuaternion, self).__init__()
        # self.tensor_args = tensor_args
        self.weight = weight
        # self.vec_weight = torch.as_tensor(vec_weight, **tensor_args)
        # self.rot_weight = self.vec_weight[0:3]
        # self.pos_weight = self.vec_weight[3:6]
        # self.position_gaussian = GaussianProjection(gaussian_params=position_gaussian_params)
        # self.orientation_gaussian = GaussianProjection(gaussian_params=orientation_gaussian_params)
        self.hinge_val = hinge_val
        self.convergence_val = convergence_val
        # self.dtype = self.tensor_args['dtype']
        # self.device = self.tensor_args['device']
        self.device = device
        self.quat_inputs = quat_inputs

    def forward(self, 
                ee_pos_batch: torch.Tensor, 
                ee_rot_batch: torch.Tensor, 
                ee_goal_pos: torch.Tensor, 
                ee_goal_rot: torch.Tensor):

        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(device=self.device)
        ee_rot_batch = ee_rot_batch.to(device=self.device)
        ee_goal_pos = ee_goal_pos.to(device=self.device)
        ee_goal_rot = ee_goal_rot.to(device=self.device)
        # st1 = time.time()
        if not self.quat_inputs:
            ee_quat_batch = matrix_to_quaternion(ee_rot_batch)
            ee_goal_quat = matrix_to_quaternion(ee_goal_rot)
        else:
            ee_quat_batch = ee_rot_batch
            ee_goal_quat = ee_goal_rot

        # print(ee_quat_batch.shape, ee_goal_quat.shape)

        # print('1', time.time()-st1)        
        # print(torch.sum(torch.isnan(ee_quat_batch)))
            
        #Translation part
        # goal_dist = torch.norm(self.pos_weight * d_g_ee, p=2, dim=-1, keepdim=True)
        # st2=time.time()
        # print(ee_pos_batch.shape, ee_goal_pos.unsqueeze(1).unsqueeze(1).shape)
        goal_disp = ee_pos_batch - ee_goal_pos.unsqueeze(1).unsqueeze(1)
        goal_dist = torch.norm(goal_disp, p=2, dim=-1)
        position_err = torch.sum(torch.square(goal_disp),dim=-1)
        # print(time.time()-st2)
        # input('...')


        #compute projection error
        # rot_err = self.I - R_g_ee
        # rot_err = torch.norm(rot_err, dim=-1)
        # rot_err_norm = torch.norm(torch.sum(self.rot_weight * rot_err,dim=-1), p=2, dim=-1, keepdim=True)
        # st3=time.time()
        # rot_err2 = torch.einsum('bijk, bk -> bij', ee_quat_batch, ee_goal_quat)

        # num_goals, batch_size, horizon, _ = ee_quat_batch.shape
        # ee_quat_batch = ee_quat_batch.view(num_goals, batch_size*horizon, -1)
        # rot_err = torch.matmul(ee_quat_batch, ee_goal_quat.unsqueeze(1).transpose(-1,-2))
        # rot_err = rot_err.view(num_goals, batch_size, horizon)
        rot_err = torch.einsum('bijk, bk -> bij', ee_quat_batch, ee_goal_quat)
        # print(time.time()-st3)

        # quat_x = (ee_goal_quat[:,0] * ee_quat_batch[:,:,0]).view(num_goals, batch_size, horizon)
        # quat_y = (ee_goal_quat[:,1] * ee_quat_batch[:,:,1]).view(num_goals, batch_size, horizon)
        # quat_z = (ee_quat_batch[:,:,2] * ee_goal_quat[:,2]).view(num_goals, batch_size, horizon)
        # quat_w = (ee_quat_batch[:,:,3] * ee_goal_quat[:,3]).view(num_goals, batch_size, horizon)

        # rot_err = quat_x + quat_y + quat_z + quat_w
        rot_err_norm = torch.norm(rot_err , p=2, dim=-1, keepdim=True)
        rot_err = 1.0 - torch.abs(rot_err)

        # rot_err = 2.0 * torch.acos(rot_err)
        #normalize to -pi,pi
        # rot_err = torch.atan2(torch.sin(rot_err), torch.cos(rot_err))
        # print(torch.sum(torch.isnan(rot_err)))
        # if(self.hinge_val > 0.0):
        #     rot_err = torch.where(goal_dist.squeeze(-1) <= self.hinge_val, rot_err, self.Z) #hard hinge

        rot_err[rot_err < self.convergence_val[0]] = 0.0
        position_err[position_err < self.convergence_val[1]] = 0.0
        # cost = self.weight[0] * self.orientation_gaussian(torch.sqrt(rot_err)) + self.weight[1] * self.position_gaussian(torch.sqrt(position_err))
        # cost = self.weight[0] * self.orientation_gaussian(rot_err) + self.weight[1] * self.position_gaussian(torch.sqrt(position_err))
        cost = self.weight[0] * rot_err + self.weight[1] * torch.sqrt(position_err)

        # dimension should be bacth * traj_length
        return cost.to(inp_device), rot_err_norm, goal_dist


