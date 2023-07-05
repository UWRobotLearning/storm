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
from torch.profiler import record_function

from ...differentiable_robot_model.coordinate_transform import matrix_to_quaternion, quaternion_to_matrix
from ..cost import DistCost, PoseCost, PoseCostQuaternion, ZeroCost, FiniteDifferenceCost
from ...mpc.rollout.arm_base import ArmBase

class ArmReacher(ArmBase):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update cfg to be kwargs
    """

    def __init__(self, cfg, device=torch.device('cpu'), dtype=torch.float32, world_params=None):
        super(ArmReacher, self).__init__(cfg=cfg,
                                         world_params=world_params,
                                         device=device,
                                         dtype=dtype)
        self.goal_state = None
        self.goal_ee_pos = None
        self.goal_ee_rot = None
        
        # device = self.tensor_args['device']
        # float_dtype = self.tensor_args['dtype']
        self.dist_cost = DistCost(**self.cfg['cost']['joint_l2'], device=self.device, dtype=self.dtype)

        # self.goal_cost = PoseCostQuaternion(**cfg['cost']['goal_pose'],
        #                                     device = self.device,
        #                                     quat_inputs=False)

        self.goal_cost = PoseCost(**cfg['cost']['goal_pose'], device = self.device)


    def compute_cost(self, state_dict=None, action_batch=None, current_state=None, no_coll=False, horizon_cost=True, return_dist=False):

        cost, state_dict = super(ArmReacher, self).compute_cost(
            state_dict = state_dict,
            action_batch = action_batch,
            current_state = current_state,
            no_coll = no_coll, 
            horizon_cost = horizon_cost)

        num_instances, curr_batch_size, num_traj_points, _ = state_dict['state_seq'].shape
        cost = cost.view(num_instances, curr_batch_size, num_traj_points)

        ee_pos_batch, ee_rot_batch = state_dict['ee_pos_seq'], state_dict['ee_rot_seq']
        # ee_pos_batch = ee_pos_batch#.view(num_instances*curr_batch_size, num_traj_points, 3)
        # ee_rot_batch = ee_rot_batch#.view(num_instances*curr_batch_size, num_traj_points, 3, 3)

        state_batch = state_dict['state_seq']#.view(num_instances*curr_batch_size, num_traj_points, -1)
        goal_ee_pos = self.goal_ee_pos
        goal_ee_rot = self.goal_ee_rot
        retract_state = self.retract_state
        goal_state = self.goal_state
        

        with record_function("pose_cost"):
            goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward(ee_pos_batch, ee_rot_batch,
                                                                        goal_ee_pos, goal_ee_rot)
        cost += goal_cost
        
        # joint l2 cost
        if self.cfg['cost']['joint_l2']['weight'] > 0.0 and goal_state is not None:
            disp_vec = state_batch[:, :,:,0:self.n_dofs] - goal_state[:,0:self.n_dofs]
            cost += self.dist_cost.forward(disp_vec)

        if return_dist:
            return cost, state_dict, rot_err_norm, goal_dist

            
        if self.cfg['cost']['zero_acc']['weight'] > 0:
            cost += self.zero_acc_cost.forward(state_batch[:, :, self.n_dofs*2:self.n_dofs*3], goal_dist=goal_dist)

        if self.cfg['cost']['zero_vel']['weight'] > 0:
            cost += self.zero_vel_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs*2], goal_dist=goal_dist)
        

        return cost, state_dict

    def compute_observations(self, state_dict=None, current_state=None):
        obs, state_dict =  super().compute_observations(state_dict, current_state)

        goal_ee_pos = self.goal_ee_pos.unsqueeze(1).unsqueeze(1)
        # goal_ee_quat = self.goal_ee_quat.unsqueeze(1).unsqueeze(1)
        goal_ee_rot = self.goal_ee_rot.unsqueeze(1).unsqueeze(1).flatten(-2,-1)
        obs = torch.cat((
            obs, 
            goal_ee_pos, goal_ee_rot,
            goal_ee_pos - state_dict['ee_pos_seq']), dim=-1)

        return obs, state_dict

    # def update_params(self, retract_state=None, goal_state=None, goal_ee_pos=None, goal_ee_rot=None, goal_ee_quat=None):
        """
        Update params for the cost terms and dynamics model.
        goal_state: n_dofs
        goal_ee_pos: 3
        goal_ee_rot: 3,3
        goal_ee_quat: 4

        """
    def update_params(self, goal, goal_state=None):

        # retract_state = goal_dict['retract_state'] if 'retract_state' in goal_dict else None
        # goal_state = goal_dict['goal_state'] if 'goal_state' in goal_dict else None
        # goal_ee_pos = goal_dict['goal_ee_pos'] if 'goal_ee_pos' in goal_dict else None
        # goal_ee_rot = goal_dict['goal_ee_rot'] if 'goal_ee_rot' in goal_dict else None
        # goal_ee_quat = goal_dict['goal_ee_quat'] if 'goal_ee_quat' in goal_dict else None
        retract_state = None
        goal_ee_pos = None
        goal_ee_quat = None
        goal_ee_rot = None
        if goal is not None:
            goal_ee_pos = goal[:, 0:3]
            goal_ee_quat = goal[:, 3:7]
            goal_ee_rot = None        
        super(ArmReacher, self).update_params(retract_state=retract_state)
        
        if goal_ee_pos is not None:
            self.goal_ee_pos = torch.as_tensor(goal_ee_pos, dtype=self.dtype, device=self.device) #.unsqueeze(0)
            self.goal_state = None
        if goal_ee_rot is not None:
            self.goal_ee_rot = torch.as_tensor(goal_ee_rot, dtype=self.dtype, device=self.device)#.unsqueeze(0)
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
            self.goal_state = None
        if goal_ee_quat is not None:
            self.goal_ee_quat = torch.as_tensor(goal_ee_quat, dtype=self.dtype, device=self.device)#.unsqueeze(0)
            self.goal_ee_rot = quaternion_to_matrix(self.goal_ee_quat)
            self.goal_state = None
        if goal_state is not None:
            self.goal_state = torch.as_tensor(goal_state, dtype=self.dtype, device=self.device)#.unsqueeze(0)
            self.goal_ee_pos, self.goal_ee_rot = self.dynamics_model.robot_model.compute_forward_kinematics(self.goal_state[:,0:self.n_dofs], self.goal_state[:,self.n_dofs:2*self.n_dofs], link_name=self.cfg['model']['ee_link_name'])
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
        
        return True
    
