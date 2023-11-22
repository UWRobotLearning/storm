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
from typing import Optional, Tuple
import torch
from storm_kit.mpc.cost import NormCost
from storm_kit.differentiable_robot_model.spatial_vector_algebra import CoordinateTransform, matrix_to_quaternion
from storm_kit.differentiable_robot_model.se3_so3_util import logMapSE3, convertQuaternionToAxisAngle

class PoseCost(NormCost):
    """ Rotation cost 

    .. math::
     
    r  &=  \sum_{i=0}^{num_rows} (R^{i,:} - R_{g}^{i,:})^2 \\
    cost &= \sum w \dot r

    
    """
    def __init__(
            self, weight, vec_weight=[], cost_type:str = 'se3_transform', 
            norm_type:str='squared_l2', device:torch.device=torch.device("cpu"), hinge_val=100.0,
            convergence_val=[0.0,0.0]):
        
        super(PoseCost, self).__init__(weight=1.0, norm_type=norm_type, device=device)

        self.cost_type = cost_type
        # self.weight = weight
        self.rot_err_weight = weight[0]
        self.trans_err_weight = weight[1]
        self.vec_weight = torch.as_tensor(vec_weight, device=self.device)
        self.trans_vec_weight = self.vec_weight[3:6]
        if self.cost_type in ['se3_transform', 'se3_twist']:
            self.rot_vec_weight = self.vec_weight[0:3]
            self.I = torch.eye(3,3,device=self.device)
            self.Z = torch.zeros(1, device=self.device)
        
        self.hinge_val = hinge_val
        self.convergence_val = convergence_val

    def forward(
            self, ee_pos_batch:torch.Tensor, ee_rot_batch:torch.Tensor, 
            ee_goal_pos:torch.Tensor, ee_goal_rot:torch.Tensor,
            jac_batch:Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        ee_pos_batch = ee_pos_batch.to(device=self.device)
        ee_rot_batch = ee_rot_batch.to(device=self.device)
        ee_goal_pos = ee_goal_pos.to(device=self.device)
        ee_goal_rot = ee_goal_rot.to(device=self.device)
        if jac_batch is not None:
            jac_batch = jac_batch.to(device=self.device)
        # if ee_pos_batch.ndim > 2:
        #     num_instances, batch_size, horizon, _ = ee_pos_batch.shape
        # else:
        #     batch_size = 1
        #     horizon = 1
        #     num_instances, _ = ee_pos_batch.shape
        
        # ee_pos_batch = ee_pos_batch.view(num_instances*batch_size*horizon, 3)
        # ee_rot_batch = ee_rot_batch.view(num_instances*batch_size*horizon, 3, 3)
        
        # if num_instances == 1:
        #     ee_goal_pos = ee_goal_pos.expand(num_instances*batch_size*horizon, 3)
        #     ee_goal_rot = ee_goal_rot.expand(num_instances*batch_size*horizon, 3,3)
        # else:
        #     ee_goal_pos = ee_goal_pos.repeat(batch_size*horizon, 1)
        #     ee_goal_rot = ee_goal_rot.repeat(batch_size*horizon, 1,1)

        # if jac_batch is not None:
        #     jac_batch = jac_batch.view(num_instances*batch_size*horizon, 6, -1)
        

        #compute error transform
        if self.cost_type in ['se3_transform', 'se3_twist']:
            ee_to_world_transform = CoordinateTransform(rot = ee_rot_batch, trans = ee_pos_batch, device=self.device)
            goal_to_world_transform = CoordinateTransform(rot = ee_goal_rot, trans = ee_goal_pos, device=self.device)
            world_to_ee_transform = ee_to_world_transform.inverse()
            goal_to_ee_transform = world_to_ee_transform.multiply_transform(goal_to_world_transform)
        
        if self.cost_type == 'se3_transform':
            # return self.forward_se3_transform(ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot)
            cost, rot_err, pos_err = self.forward_se3_transform(goal_to_ee_transform)
        elif self.cost_type == 'se3_twist':
            # return self.forward_se3_twist(ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot)
            cost, rot_err, pos_err = self.forward_se3_twist(goal_to_ee_transform, jac_batch)
        elif self.cost_type == 'quaternion':
            cost, rot_err, pos_err = self.forward_quaternion(ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot)
        
        # cost = cost.view(num_instances, batch_size, horizon)
        # rot_err = rot_err.view(num_instances, batch_size, horizon)
        # pos_err = pos_err.view(num_instances, batch_size, horizon)
        # rot_err_norm = rot_err_norm.view(num_instances, batch_size, horizon)
        # goal_dist = goal_dist.view(num_instances, batch_size, horizon)

        return cost, rot_err, pos_err
        


    # def forward_se3_transform(
    #         self, ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot):

    def forward_se3_transform(
            self, goal_to_ee_transform:CoordinateTransform)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # ee_pos_batch = ee_pos_batch.to(device=self.device)
        # ee_rot_batch = ee_rot_batch.to(device=self.device)
        # ee_goal_pos = ee_goal_pos.to(device=self.device)
        # ee_goal_rot = ee_goal_rot.to(device=self.device)

        #Inverse of goal transform
        # R_g_t = ee_goal_rot.transpose(-2,-1) # w_R_g -> g_R_w
        # R_g_t_d = (-1.0 * R_g_t @ ee_goal_pos.t()).transpose(-2,-1) # -g_R_w * w_d_g -> g_d_g
        # #Rotation part
        # R_g_ee = R_g_t @ ee_rot_batch # g_R_w * w_R_ee -> g_R_ee
        
        # #Translation part
        # # transpose is done for matmul
        # term1 = (R_g_t @ ee_pos_batch.transpose(-2,-1)).transpose(-2,-1) # g_R_w * w_d_ee -> g_d_ee
        # d_g_ee = term1 + R_g_t_d # g_d_g + g_d_ee
        # d_g_ee, R_g_ee = self.compute_error_transform(ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot)
        goal_trans_ee = goal_to_ee_transform.translation()
        goal_rot_ee = goal_to_ee_transform.rotation()

        #compute translation error
        position_err = super().forward(self.trans_vec_weight * goal_trans_ee, keepdim=False)
        # goal_dist = torch.norm(self.pos_weight * d_g_ee, p=2, dim=-1, keepdim=True)
        
        # position_err = (torch.sum(torch.square(self.pos_weight * d_g_ee),dim=-1))
        
        #compute rotation projection error
        rotation_err = self.I - goal_rot_ee
        rotation_err = torch.square(rotation_err)
        rotation_err = torch.sum(rotation_err, dim=-1)
        rotation_err = torch.sum(self.rot_vec_weight * rotation_err, dim=-1)
        # rot_err = torch.norm(rot_err, dim=-1)
        # rot_err_norm = torch.norm(torch.sum(self.rot_vec_weight * rot_err,dim=-1), p=2, dim=-1, keepdim=True)
        
        # rot_err = torch.square(torch.sum(self.rot_vec_weight * rot_err, dim=-1))


        if self.hinge_val > 0.0:
            rotation_err = torch.where(position_err.squeeze(-1) <= self.hinge_val, rotation_err, self.Z) #hard hinge

        rotation_err[rotation_err < self.convergence_val[0]] = 0.0
        position_err[position_err < self.convergence_val[1]] = 0.0
        # cost = self.rot_err_weight * torch.sqrt(rotation_err) + self.trans_err_weight * torch.sqrt(position_err)
        cost = self.rot_err_weight * rotation_err + self.trans_err_weight * position_err
        
        # dimension should be bacth * traj_length
        return cost, rotation_err, position_err


    def forward_se3_twist(
            self, goal_to_ee_transform:CoordinateTransform, jac_batch:Optional[torch.Tensor]=None)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        _, omega, v = logMapSE3(goal_to_ee_transform.rotation(), goal_to_ee_transform.translation())

        position_err = super().forward(self.trans_vec_weight * v, keepdim=False)
        rotation_err = super().forward(self.rot_vec_weight * omega, keepdim=False)
        rotation_err[rotation_err < self.convergence_val[0]] = 0.0
        position_err[position_err < self.convergence_val[1]] = 0.0

        if jac_batch is not None:
            #Compute jacobian transpose projection
            jac_T = jac_batch.transpose(-2,-1)
            ee_twist = torch.cat([v, omega], dim=-1)[:, :, None]
            q_vel_des = (jac_T @ ee_twist).squeeze(-1)
            cost = 500.0 * super().forward(q_vel_des, keepdim=False)

        else:
            cost = self.rot_err_weight * rotation_err + self.trans_err_weight * position_err
        
        return cost, rotation_err, position_err


    def forward_quaternion(
            self, ee_pos_batch: torch.Tensor, ee_rot_batch: torch.Tensor, 
            ee_goal_pos: torch.Tensor, ee_goal_rot: torch.Tensor):

        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(device=self.device)
        ee_rot_batch = ee_rot_batch.to(device=self.device)
        ee_goal_pos = ee_goal_pos.to(device=self.device)
        ee_goal_rot = ee_goal_rot.to(device=self.device)
        ee_quat_batch = matrix_to_quaternion(ee_rot_batch)
        ee_goal_quat = matrix_to_quaternion(ee_goal_rot)

        goal_disp = ee_pos_batch - ee_goal_pos
        position_err = super().forward(self.trans_vec_weight * goal_disp, keepdim=False) #torch.norm(goal_disp, p=2, dim=-1)

        #quaternion error
        conj_quat = ee_quat_batch
        conj_quat[..., 1:] *= -1.0

        quat_res = quat_multiply(ee_goal_quat, conj_quat)
        # quat_res = -1.0 * quat_res * torch.sign(quat_res[..., 0]).unsqueeze(-1)
        # quat_res[..., 0] = 0.0

        rotation_err = super().forward(quat_res[..., 1:], keepdim=False)

        # rot_err = torch.einsum('bijk, bk -> bij', ee_quat_batch, ee_goal_quat)

        # # rot_err = quat_x + quat_y + quat_z + quat_w
        # rot_err_norm = torch.norm(rot_err , p=2, dim=-1, keepdim=True)
        # rot_err = 1.0 - torch.abs(rot_err)

        # rot_err = 2.0 * torch.acos(rot_err)
        #normalize to -pi,pi
        # rot_err = torch.atan2(torch.sin(rot_err), torch.cos(rot_err))
        # print(torch.sum(torch.isnan(rot_err)))
        # if(self.hinge_val > 0.0):
        #     rot_err = torch.where(goal_dist.squeeze(-1) <= self.hinge_val, rot_err, self.Z) #hard hinge

        rotation_err[rotation_err < self.convergence_val[0]] = 0.0
        position_err[position_err < self.convergence_val[1]] = 0.0
        # cost = self.weight[0] * self.orientation_gaussian(torch.sqrt(rot_err)) + self.weight[1] * self.position_gaussian(torch.sqrt(position_err))
        # cost = self.weight[0] * self.orientation_gaussian(rot_err) + self.weight[1] * self.position_gaussian(torch.sqrt(position_err))
        cost = self.rot_err_weight * rotation_err + self.trans_err_weight * torch.sqrt(position_err)

        return cost.to(inp_device), rotation_err, position_err


def quat_multiply(q1:torch.Tensor, q2:torch.Tensor) -> torch.Tensor:
    a_w = q1[..., 0]
    a_x = q1[..., 1]
    a_y = q1[..., 2]
    a_z = q1[..., 3]
    b_w = q2[..., 0]
    b_x = q2[..., 1]
    b_y = q2[..., 2]
    b_z = q2[..., 3]


    q_res_w = (a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z).unsqueeze(-1)
    q_res_x = (a_w * b_x + b_w * a_x + a_y * b_z - b_y * a_z).unsqueeze(-1)
    q_res_y = (a_w * b_y + b_w * a_y + a_z * b_x - b_z * a_x).unsqueeze(-1)
    q_res_z = (a_w * b_z + b_w * a_z + a_x * b_y - b_x * a_y).unsqueeze(-1)

    return torch.cat([q_res_w, q_res_x, q_res_y, q_res_z], dim=-1)



    # def compute_error_transform(
    #         self, ee_pos_world: torch.Tensor, ee_rot_world: torch.Tensor,
    #         goal_pos_world: torch.Tensor, goal_rot_world: torch.Tensor):
        
    #     # Given T_e^W (end effector in world) and T_g^W (goal in world)
    #     # compute T_g^e (goal in end effector frame) using
    #     # T_g^e = (T_e^W)^-1 * T_g^W . .  
    #     #Inverse of ee transform
    #     ee_rot_world_T = ee_rot_world.transpose(-2,-1) # w_R_g -> g_R_w
    #     #Rotation part
    #     goal_rot_ee = ee_rot_world_T @ goal_rot_world # g_R_w * w_R_ee -> g_R_ee
    #     #Translation part
    #     pos_diff = goal_pos_world - ee_pos_world
    #     goal_pos_ee = (ee_rot_world_T @ pos_diff.unsqueeze(-1)).squeeze(-1)#.transpose(-2,-1)

    #     return goal_pos_ee, goal_rot_ee
