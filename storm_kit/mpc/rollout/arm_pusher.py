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
from typing import Dict, Optional, Tuple
import torch
from torch.profiler import record_function

from ...differentiable_robot_model.spatial_vector_algebra import matrix_to_quaternion, quaternion_to_matrix
from ..cost import DistCost, PoseCost
from ...mpc.rollout.arm_base import ArmBase

class ArmPusher(ArmBase):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update cfg to be kwargs
    """

    def __init__(self, cfg, world_params=None, value_function=None, viz_rollouts=False, device=torch.device('cpu')):
        super(ArmPusher, self).__init__(cfg=cfg,
                                         world_params=world_params,
                                         value_function=value_function,
                                         viz_rollouts=viz_rollouts,
                                         device=device,
                                         dtype=torch.float32)
        self.goal_state = None
        self.goal_ee_pos = None
        self.goal_ee_rot = None
        
        self.dist_cost = DistCost(**self.cfg['cost']['joint_l2'], device=self.device)

        # self.goal_cost = PoseCostQuaternion(**cfg['cost']['goal_pose'],
        #                                     device = self.device,
        #                                     quat_inputs=False)

        self.goal_cost = PoseCost(**cfg['cost']['goal_pose'], device=self.device)
        self.task_specs = cfg['task_specs']
        self.default_ee_goal = torch.tensor(self.task_specs['default_ee_target'], device=self.device)
        self.init_buffers()

    def init_buffers(self):
        # self.ee_goal_buff = torch.zeros(self.num_instances, 7, device=self.device)
        self.object_goal_buff = torch.zeros(self.num_instances, 2, device=self.device)
        self.object_init_pos_buff = torch.zeros(self.num_instances, 2, device=self.device)
        self.prev_state_buff = torch.zeros(self.num_instances, 10, 22, device=self.device)

    def compute_cost(
            self, 
            state_dict: Dict[str, torch.Tensor], 
            action_batch: Optional[torch.Tensor]=None,
            termination: Optional[Dict[str, torch.Tensor]]=None, 
            horizon_cost: bool=True, return_dist:bool=False):

        cost, cost_terms, state_dict = super(ArmPusher, self).compute_cost(
            state_dict = state_dict,
            action_batch = action_batch,
            termination = termination,
            horizon_cost = horizon_cost)

        # num_instances, curr_batch_size, num_traj_points, _ = state_dict['state_seq'].shape
        # cost = cost.view(num_instances, curr_batch_size, num_traj_points)

        ee_pos_batch, ee_rot_batch = state_dict['ee_pos_seq'], state_dict['ee_rot_seq']
        # ee_pos_batch = ee_pos_batch#.view(num_instances*curr_batch_size, num_traj_points, 3)
        # ee_rot_batch = ee_rot_batch#.view(num_instances*curr_batch_size, num_traj_points, 3, 3)

        state_batch = state_dict['state_seq']#.view(num_instances*curr_batch_size, num_traj_points, -1)
        goal_ee_pos = self.goal_ee_pos
        goal_ee_rot = self.goal_ee_rot
        goal_state = self.goal_state
        
        # with record_function("pose_cost"):
        #     goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward(ee_pos_batch, ee_rot_batch,
        #                                                                 goal_ee_pos, goal_ee_rot)

        with record_function("pose_cost"):
            goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward(ee_pos_batch, ee_rot_batch,
                                                                        goal_ee_pos, goal_ee_rot)
            # goal_cost[:,:,0:-1] = 0.
        cost += goal_cost
                
        return cost, cost_terms, state_dict


    def compute_termination(self, state_dict: Dict[str,torch.Tensor], act_batch: torch.Tensor):
        return super().compute_termination(state_dict, act_batch)


    def compute_metrics(self, episode_data: Dict[str, torch.Tensor]):
        return {'error': 0.0}


    def reset(self):
        env_ids = torch.arange(self.num_instances, device=self.device)
        return self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        goal_position_noise = self.task_specs['target_position_noise']
        goal_rotation_noise = self.task_specs['target_rotation_noise']

        self.ee_goal_buff[env_ids] = self.default_ee_goal

        if goal_position_noise > 0.:
            #randomize goal position around the default
            self.ee_goal_buff[env_ids, 0] = self.ee_goal_buff[env_ids, 0] +  2.0*goal_position_noise * (torch.rand_like(self.ee_goal_buff[env_ids, 0], device=self.device) - 0.5)
            self.ee_goal_buff[env_ids, 1] = self.ee_goal_buff[env_ids, 1] +  2.0*goal_position_noise * (torch.rand_like(self.ee_goal_buff[env_ids, 1], device=self.device) - 0.5)
            self.ee_goal_buff[env_ids, 2] = self.ee_goal_buff[env_ids, 2] +  2.0*goal_position_noise * (torch.rand_like(self.ee_goal_buff[env_ids, 2], device=self.device) - 0.5)
        
        if goal_rotation_noise > 0.:
            #randomize goal orientation
            raise NotImplementedError('oritentation randomization not implemented')

        self.goal_ee_pos = self.ee_goal_buff[..., 0:3]
        self.goal_ee_quat = self.ee_goal_buff[..., 3:7]
        self.goal_ee_rot = quaternion_to_matrix(self.goal_ee_quat)

        self.prev_state_buff[env_ids] = torch.zeros_like(self.prev_state_buff[env_ids])


        # #reset EE target 
        # if randomization_mode == "fixed":
        #     pass

        # if randomization_mode in ["randomize_position", "randomize_full_pose"]:
        #     #randomize position
        #     curr_target_buff[env_ids, 0] = 0.2 + 0.4 * torch.rand(
        #         size=(env_ids.shape[0],), device=self.device) #x from [0.2, 0.6)
        #     curr_target_buff[env_ids, 1] = -0.3 + 0.6 * torch.rand(
        #         size=(env_ids.shape[0],), device=self.device) #y from [-0.3, 0.3)
        #     curr_target_buff[env_ids, 2] = 0.2 + 0.3 * torch.rand(
        #         size=(env_ids.shape[0],), device=self.device) #z from [0.2, 0.5)

        
        # if randomization_mode == "randomize_full_pose":
        #     #randomize orientation
        #     roll = -torch.pi + 2*torch.pi * torch.rand(
        #         size=(env_ids.shape[0],1), device=self.device) #roll from [-pi, pi)
        #     pitch = -torch.pi + 2*torch.pi * torch.rand(
        #         size=(env_ids.shape[0],1), device=self.device) #pitch from [-pi, pi)
        #     yaw = -torch.pi + 2*torch.pi * torch.rand(
        #         size=(env_ids.shape[0],1), device=self.device) #yaw from [-pi, pi)
        #     quat = matrix_to_quaternion(rpy_angles_to_matrix(torch.cat([roll, pitch, yaw], dim=-1)))
        #     curr_target_buff[env_ids, 3:7] = quat
        
        reset_data = {}
        goal_dict = dict(ee_goal=self.ee_goal_buff)
        reset_data['goal_dict'] = goal_dict

        self.dynamics_model.reset_idx(env_ids)

        return reset_data


    # def compute_cost(self, state_dict=None, action_batch=None, current_state=None, no_coll=False, horizon_cost=True, return_dist=False):

    #     cost, state_dict = super(ArmReacher, self).compute_cost(
    #         state_dict = state_dict,
    #         action_batch = action_batch,
    #         current_state = current_state,
    #         no_coll = no_coll, 
    #         horizon_cost = horizon_cost)

    #     num_instances, curr_batch_size, num_traj_points, _ = state_dict['state_seq'].shape
    #     cost = cost.view(num_instances, curr_batch_size, num_traj_points)

    #     ee_pos_batch, ee_rot_batch = state_dict['ee_pos_seq'], state_dict['ee_rot_seq']
    #     # ee_pos_batch = ee_pos_batch#.view(num_instances*curr_batch_size, num_traj_points, 3)
    #     # ee_rot_batch = ee_rot_batch#.view(num_instances*curr_batch_size, num_traj_points, 3, 3)

    #     state_batch = state_dict['state_seq']#.view(num_instances*curr_batch_size, num_traj_points, -1)
    #     goal_ee_pos = self.goal_ee_pos
    #     goal_ee_rot = self.goal_ee_rot
    #     retract_state = self.retract_state
    #     goal_state = self.goal_state
        

    #     # with record_function("pose_cost"):
    #     #     goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward(ee_pos_batch, ee_rot_batch,
    #     #                                                                 goal_ee_pos, goal_ee_rot)

    #     with record_function("pose_cost"):
    #         goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward(ee_pos_batch, ee_rot_batch,
    #                                                                     goal_ee_pos, goal_ee_rot)
    #         # goal_cost[:,:,0:-1] = 0.
    #     cost += goal_cost
        
    #     # joint l2 cost
    #     if self.cfg['cost']['joint_l2']['weight'] > 0.0 and goal_state is not None:
    #         disp_vec = state_batch[:,:,:,0:self.n_dofs] - goal_state[:,0:self.n_dofs]
    #         dist_cost, _ = self.dist_cost.forward(disp_vec)
    #         cost += dist_cost #self.dist_cost.forward(disp_vec)

    #     if return_dist:
    #         return cost, state_dict, rot_err_norm, goal_dist

    #     if self.cfg['cost']['zero_acc']['weight'] > 0:
    #         cost += self.zero_acc_cost.forward(state_batch[:, :, self.n_dofs*2:self.n_dofs*3], goal_dist=goal_dist)

    #     if self.cfg['cost']['zero_vel']['weight'] > 0:
    #         cost += self.zero_vel_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs*2], goal_dist=goal_dist)
        

    #     return cost, state_dict

    def compute_observations(self, state_dict: Dict[str, torch.Tensor]):
        obs, state_dict =  super().compute_observations(state_dict)

        # goal_ee_pos = self.goal_ee_pos.unsqueeze(1).unsqueeze(1)
        # # goal_ee_quat = self.goal_ee_quat.unsqueeze(1).unsqueeze(1)
        # goal_ee_rot = self.goal_ee_rot.unsqueeze(1).unsqueeze(1).flatten(-2,-1)
        
        # obs = torch.cat((
        #     obs, 
        #     goal_ee_pos, goal_ee_rot,
        #     goal_ee_pos - state_dict['ee_pos_seq']), dim=-1)
        obs = torch.cat((
            obs, 
            self.ee_goal_buff[:, 0:3],
            state_dict['ee_pos_seq'],
            self.ee_goal_buff[:, 0:3] - state_dict['ee_pos_seq']), dim=-1)

        return obs, state_dict
    
    def update_params(self, param_dict):
        goal_dict = param_dict['goal_dict']
        retract_state = goal_dict['retract_state'] if 'retract_state' in goal_dict else None
        ee_goal = goal_dict['ee_goal'] if 'ee_goal' in goal_dict else None
        joint_goal = goal_dict['joint_goal'] if 'joint_goal' in goal_dict else None
        
        # goal_ee_pos = goal_dict['goal_ee_pos'] if 'goal_ee_pos' in goal_dict else None
        # goal_ee_rot = goal_dict['goal_ee_rot'] if 'goal_ee_rot' in goal_dict else None
        # goal_ee_quat = goal_dict['goal_ee_quat'] if 'goal_ee_quat' in goal_dict else None
        # retract_state = None
        # goal_ee_pos = None
        # goal_ee_quat = None
        # goal_ee_rot = None

        if ee_goal is not None:
            goal_ee_pos = ee_goal[:, 0:3]
            goal_ee_quat = ee_goal[:, 3:7]
            goal_ee_rot = None        
        
        super(ArmPusher, self).update_params(retract_state=retract_state)
        
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
        if joint_goal is not None:
            self.goal_state = torch.as_tensor(joint_goal, dtype=self.dtype, device=self.device)#.unsqueeze(0)
            self.goal_ee_pos, self.goal_ee_rot = self.dynamics_model.robot_model.compute_forward_kinematics(self.goal_state[:,0:self.n_dofs], self.goal_state[:,self.n_dofs:2*self.n_dofs], link_name=self.cfg['model']['ee_link_name'])
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
        return True

    @property
    def obs_dim(self)->int:
        return 23

    @property
    def action_dim(self)->int:
        return self.n_dofs

    @property
    def action_lims(self)->Tuple[torch.Tensor, torch.Tensor]:
        act_highs = torch.tensor([self.cfg['model']['max_acc']] * self.action_dim,  device=self.device)
        act_lows = torch.tensor([-1.0 * self.cfg['model']['max_acc']] * self.action_dim, device=self.device)
        return act_lows, act_highs
