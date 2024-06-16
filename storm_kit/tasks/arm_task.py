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
from typing import Optional, Dict, Tuple, Union, List
from torch.profiler import record_function
import time
from operator import mul
from functools import reduce


from storm_kit.util_file import join_path, get_assets_path
from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.spatial_vector_algebra import matrix_to_quaternion, quaternion_to_matrix
from storm_kit.mpc.cost import NormCost, StopCost, FiniteDifferenceCost 
from storm_kit.mpc.cost.bound_cost import BoundCost
from storm_kit.mpc.cost.manipulability_cost import ManipulabilityCost
from storm_kit.mpc.cost import CollisionCost, PrimitiveCollisionCost #VoxelCollisionCost
from storm_kit.mpc.cost.robot_self_collision_cost import RobotSelfCollisionCost
from storm_kit.mpc.model.integration_utils import build_fd_matrix
import torch
import torch.nn as nn

class ArmTask(nn.Module):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update cfg to be kwargs
    """
    dt_traj_params: Optional[Dict[str, float]]
    max_acc: float
    max_jerk: float
    robot_default_dof_pos:torch.Tensor
    retract_state:torch.Tensor
    joint_lim_dicts:List[Dict[str, float]]
    
    def __init__(self, cfg, world_cfg, viz_rollouts:bool=False, device=torch.device('cpu')):
        super().__init__()
        self.cfg = cfg
        self.viz_rollout:bool = viz_rollouts
        self.batch_size:int = cfg.get('batch_size', 1)
        self.horizon:int = cfg.get('horizon', 1)
        self.dt_traj_params:Dict[str, float] = cfg.get('dt_traj_params', None)
        self.max_acc:float = cfg['max_acc']
        self.max_jerk:float = cfg['max_jerk']
        self.robot_cfg = cfg['robot']
        self.ee_link_name:str = self.robot_cfg['ee_link_name']
        self.device:torch.device = device

        self.robot_model = torch.jit.script(DifferentiableRobotModel(self.robot_cfg, device=self.device))
        self.n_dofs:int = self.robot_model._n_dofs
        self.robot_default_dof_pos:torch.Tensor = torch.as_tensor(self.robot_cfg['default_dof_pos'], device=self.device).unsqueeze(0)


        # self.n_links = self.robot_model.n_links
        # self.link_names = cfg.get('robot_link_names')
        # if self.link_names is None:
        #     self.link_names = cfg['model'].get('link_names')
        # # self.num_links = len(self.link_names)
                        
        # self.jacobian_cost = JacobianCost(ndofs=self.n_dofs, device=device,
        #                                   float_dtype=float_dtype,
        #                                   retract_weight=cfg['cost']['retract_weight'])
        
        # self.null_cost = ProjectedDistCost(ndofs=self.n_dofs, device=device, 
        #                                    **cfg['cost']['null_space'])
        
        self.manipulability_cost = torch.jit.script(ManipulabilityCost(device=self.device, **cfg['cost']['manipulability']))
        self.zero_q_vel_cost = torch.jit.script(NormCost(**self.cfg['cost']['zero_q_vel'], device=self.device))
        self.zero_q_acc_cost = torch.jit.script(NormCost(**self.cfg['cost']['zero_q_acc'], device=self.device))
        self.zero_q_jerk_cost = torch.jit.script(NormCost(**self.cfg['cost']['zero_q_jerk'], device=self.device))
        self.ee_vel_twist_cost = torch.jit.script(NormCost(**self.cfg['cost']['ee_vel_twist'], device=self.device))
        self.ee_acc_twist_cost = torch.jit.script(NormCost(**self.cfg['cost']['ee_acc_twist'], device=self.device))

        if self.dt_traj_params is not None:
            if self.cfg['cost']['stop_cost']['weight'] > 0:
                self.stop_cost = torch.jit.script(
                    StopCost(**cfg['cost']['stop_cost'],
                    horizon=self.horizon,
                    device=self.device,
                    dt_traj_params=self.dt_traj_params))
            
            if self.cfg['cost']['stop_cost_acc']['weight'] > 0:        
                self.stop_cost_acc = torch.jit.script(
                    StopCost(**cfg['cost']['stop_cost_acc'],
                    horizon=self.horizon,
                    device=self.device,
                    dt_traj_params=self.dt_traj_params))

        self.retract_state:torch.Tensor = torch.tensor([self.cfg['cost']['retract_state']], device=device)

        if self.cfg['cost']['smooth_cost']['weight'] > 0:
            self.fd_matrix:torch.Tensor = build_fd_matrix(10 - self.cfg['cost']['smooth_cost']['order'] + 1, device=self.device, order=self.cfg['cost']['smooth_cost']['order'])
            self.smooth_cost = FiniteDifferenceCost(
                **self.cfg['cost']['smooth_cost'], 
                horizon = self.horizon + 1,
                device=self.device)

        self.primitive_collision_cost = PrimitiveCollisionCost(
            world_cfg=world_cfg, # robot_collision_params=cfg.robot_collision_params,
            world_collision_params=cfg.world_collision_params, 
            batch_size= self.batch_size * self.horizon, #self.num_instances * 
            device=self.device, **self.cfg['cost']['primitive_collision'])
        
        # self.primitive_collision_cost = torch.compile(PrimitiveCollisionCost(
        #     world_cfg=world_cfg, # robot_collision_params=cfg.robot_collision_params,
        #     world_collision_params=cfg.world_collision_params, 
        #     batch_size= self.batch_size * self.horizon, #self.num_instances * 
        #     device=self.device, **self.cfg['cost']['primitive_collision']))

        # if cfg['cost']['robot_self_collision']['weight'] > 0.0:
        # self.robot_self_collision_cost = RobotSelfCollisionCost(
        #     config=model_params['robot_collision_params'], batch_size=self.num_instances * self.batch_size * self.horizon,
        #     device=self.device, **self.cfg['cost']['robot_self_collision'])

        #Initialize bound cost
        self.joint_lim_dicts:List[Dict[str, float]] = self.robot_model.get_joint_limits()
        self.state_upper_bounds:torch.Tensor = torch.zeros(3*self.n_dofs, device=self.device)
        self.state_lower_bounds:torch.Tensor = torch.zeros(3*self.n_dofs, device=self.device)
        
        for i in range(self.n_dofs):
            self.state_upper_bounds[i] = self.joint_lim_dicts[i]['upper']
            self.state_lower_bounds[i] = self.joint_lim_dicts[i]['lower']
            self.state_upper_bounds[i+self.n_dofs] = self.joint_lim_dicts[i]['velocity'] #* vel_scale
            self.state_lower_bounds[i+self.n_dofs] = -1.0 * self.joint_lim_dicts[i]['velocity'] #* vel_scale
            self.state_upper_bounds[i+2*self.n_dofs] = self.max_acc
            self.state_lower_bounds[i+2*self.n_dofs] = -1.0 * self.max_acc

        self.bounds:torch.Tensor = torch.cat([
            self.state_lower_bounds.unsqueeze(0), 
            self.state_upper_bounds.unsqueeze(0)], dim=0).T
        
        self.bound_cost = torch.jit.script(
            BoundCost(**cfg['cost']['state_bound'],
            bounds=self.bounds, device=self.device))

        # self.link_pos_seq = torch.zeros((self.num_instances, self.num_links, 3), device=self.device)
        # self.link_rot_seq = torch.zeros((self.num_instances, self.num_links, 3, 3), device=self.device)

        self.vis_initialized:bool = False
        # self.full_state_dict = None
    
    def forward(self, state_dict: Dict[str, torch.Tensor], act_batch:Optional[torch.Tensor]=None):
        state_dict = self.compute_full_state(state_dict)
        
        termination, termination_cost, term_info = self.compute_termination(state_dict, act_batch)
        cost, cost_terms = self.compute_cost(state_dict, act_batch)
        obs = self.compute_observations(state_dict)


        if termination_cost is not None:
            cost += termination_cost
            cost_terms['termination'] = termination_cost

        return obs, cost, termination, cost_terms, termination_cost, term_info
    
        
    def compute_cost(
            self, 
            state_dict: Dict[str, torch.Tensor], 
            action_batch: Optional[torch.Tensor]=None):

        cost_terms = {}
        # state_dict = self.compute_full_state(state_dict)

        q_pos_batch = state_dict['q_pos']
        q_vel_batch = state_dict['q_vel']
        q_acc_batch = state_dict['q_acc']
        orig_size = q_pos_batch.size()[0:-1]


        # action_batch = action_batch.view(self.num_instances * self.batch_size, self.horizon, -1)
        # ee_jacobian = state_dict['ee_jacobian'].view(self.num_instances*self.batch_size, self.horizon, 6, -1)
        # ee_vel_twist_batch = state_dict['ee_vel_twist'].view(self.num_instances*self.batch_size, self.horizon, -1)
        # ee_acc_twist_batch = state_dict['ee_acc_twist'].view(self.num_instances*self.batch_size, self.horizon, -1)
        ee_jacobian = state_dict['ee_jacobian'].view(-1, 6, self.n_dofs)
        ee_vel_twist_batch = state_dict['ee_vel_twist']
        ee_acc_twist_batch = state_dict['ee_acc_twist']

        # lin_jac_batch, ang_jac_batch = state_dict['lin_jac_seq'], state_dict['ang_jac_seq']
        # lin_jac_batch = lin_jac_batch.view(self.num_instances*self.batch_size, self.horizon, 3, self.n_dofs)
        # ang_jac_batch = ang_jac_batch.view(self.num_instances*self.batch_size, self.horizon, 3, self.n_dofs)

        # link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        # link_pos_batch = link_pos_batch.view(self.num_instances*self.batch_size, self.horizon, self.num_links, 3)
        # link_rot_batch = link_rot_batch.view(self.num_instances*self.batch_size, self.horizon, self.num_links, 3, 3)

        # prev_state = state_dict['prev_state_seq']
        # prev_state = prev_state.view(self.num_instances*prev_state.shape[-2], prev_state.shape[-1])
        # prev_state_tstep = prev_state[:,-1]

        # retract_state = self.retract_state
        # J_full = torch.cat((lin_jac_batch, ang_jac_batch), dim=-2)
        
        #null-space cost
        # if self.cfg['cost']['null_space']['weight'] > 0:
        # null_disp_cost, _ = self.null_cost.forward(
        #     state_batch[:,:,0:self.n_dofs] -
        #     retract_state[:,0:self.n_dofs],
        #     ee_jacobian,
        #     proj_type='identity',
        #     dist_type='squared_l2')
        
        # cost = null_disp_cost
        
        # if self.cfg['cost']['manipulability']['weight'] > 0.0:
        with record_function('manipulability_cost'):
            manip_cost, manip_cost_info = self.manipulability_cost.forward(ee_jacobian)
            cost_terms['manip_score'] = manip_cost_info['manip_score']
            cost_terms['manip_cost'] = manip_cost
            cost = manip_cost.view(orig_size)

        if self.cfg['cost']['zero_q_vel']['weight'] > 0:
            with record_function('zero_q_vel_cost'):
                cost += self.zero_q_vel_cost.forward(q_vel_batch) #.view(-1,self.n_dofs) #.view(self.num_instances * self.batch_size, self.horizon)

        if self.cfg['cost']['zero_q_acc']['weight'] > 0:
            with record_function('zero_q_acc_cost'):
                cost += self.zero_q_acc_cost.forward(q_acc_batch)#.view(self.num_instances * self.batch_size, self.horizon)

        if self.cfg['cost']['zero_q_jerk']['weight'] > 0:
            with record_function('zero_q_jerk_cost'):
                q_jerk_batch = state_dict['q_jerk']
                cost += self.zero_q_jerk_cost.forward(q_jerk_batch)

        if self.cfg['cost']['ee_vel_twist']['weight'] > 0:
            with record_function('ee_vel_twist_cost'):
                cost += self.ee_vel_twist_cost.forward(ee_vel_twist_batch)#.view(self.num_instances * self.batch_size, self.horizon)

        if self.cfg['cost']['ee_acc_twist']['weight'] > 0:
            with record_function('ee_acc_twist_cost'):
                cost += self.ee_acc_twist_cost.forward(ee_acc_twist_batch)#.view(self.num_instances * self.batch_size, self.horizon)

        #TODO: This must also be MPC specific?
        # if self.cfg['cost']['smooth_cost']['weight'] > 0:
        #     with record_function('smooth_cost'):
        #         order = self.cfg['cost']['smooth_cost']['order']
        #         n_mul = 2 #TODO: This must be decided based on order and control space
        #         # prev_action = state_dict['prev_action']
        #         # prev_action = prev_action.unsqueeze(1).expand(self.num_instances*self.batch_size, 1, -1)
        #         # act_buff = torch.cat([prev_action, action_batch], dim=-2)
        #         state = state_batch[:,:, n_mul * self.n_dofs : (n_mul+1) * self.n_dofs]
        #         p_state = prev_state[-order:, n_mul*self.n_dofs: (n_mul+1) * self.n_dofs].unsqueeze(0)
        #         p_state = p_state.expand(state.shape[0], -1, -1)
        #         state_buffer = torch.cat((p_state, state), dim=1)
        #         prev_dt = (self.fd_matrix @ prev_state_tstep)[-order:]
        #         traj_dt = torch.cat((prev_dt, self.traj_dt[0:-1]))
        #         cost += self.smooth_cost.forward(state_buffer, traj_dt.unsqueeze(-1))

        if self.dt_traj_params is not None:
            if self.cfg['cost']['stop_cost']['weight'] > 0:
                with record_function("stop_cost"):
                    cost += self.stop_cost.forward(q_vel_batch) #.view(new_size)

            if self.cfg['cost']['stop_cost_acc']['weight'] > 0:
                with record_function("stop_cost_acc"):
                    # q_acc_batch = q_acc_batch.view(self.batch_size, self.horizon, -1)
                    cost += self.stop_cost_acc.forward(q_acc_batch) #.view(new_size)

        # cost = cost.view(orig_size)

        return cost, cost_terms

    def compute_observations(
            self, 
            state_dict: Dict[str,torch.Tensor],
            cost_terms: Optional[Dict[str, torch.Tensor]]=None):
        # import pdb; pdb.set_trace()
        q_pos = state_dict['q_pos']
        q_vel = state_dict['q_vel']
        q_acc = state_dict['q_acc']
        ee_pos = state_dict['ee_pos']
        ee_rot = state_dict['ee_rot']
        ee_quat = state_dict['ee_quat']
        ee_vel_twist = state_dict['ee_vel_twist']
        ee_acc_twist = state_dict['ee_acc_twist']
        # ee_rot_obs = ee_rot_batch[..., 0:2].flatten(-2,-1)

        # if 'ee_quat_seq' not in state_dict:
        #     ee_rot_batch = state_dict['ee_rot_seq']
        #     ee_quat_batch = matrix_to_quaternion(ee_rot_batch.view(new_size, 3, 3)).view(*orig_size, -1)
        #     state_dict['ee_quat_seq'] = ee_quat_batch
        # else:
        #     ee_quat_batch = state_dict['ee_quat_seq']
        
        # if cost_terms is None:
        #     state_batch = torch.cat((q_pos_batch, q_vel_batch, q_acc_batch), dim=-1).view(new_size, -1)
        #     _, bound_cost_info = self.bound_cost.forward(state_batch)
        #     bound_dist = bound_cost_info['bound_dist'].view(*orig_size,-1)
        # else:
        #     bound_dist = cost_terms['bound_dist']
        # import pdb; pdb.set_trace()

        ######all ee obs#######
        obs = torch.cat(
            (ee_pos, ee_rot.flatten(-2,-1), ee_vel_twist, ee_acc_twist), dim=-1)

        # obs = torch.cat(
        #             (ee_vel_twist, ee_acc_twist), dim=-1)

        # import pdb; pdb.set_trace()
        # gravity_vector = torch.tensor([0, 0, -9.81], device=self.device).float().expand(ee_rot.size(0), -1)        
        # rotated_gravity_vector = torch.bmm(ee_rot, gravity_vector.unsqueeze(2)).squeeze(2)
        #rotated gravity vector
        # gravity_vector = torch.tensor([0, 0, -9.81], device=ee_rot.device).float().expand(ee_rot.size(0), -1)
        # g_ee = torch.einsum('...ij,...j->...i', ee_rot, gravity_vector)

        # #surface normal
        # normal_vector = ee_rot[..., 2]  #z axis of the ee should be the normal
        # obs = torch.cat(
        #     (normal_vector,), dim=-1) # ,  q_pos, q_vel, 
        # import pdb; pdb.set_trace()
        
        return obs

    @torch.jit.export
    def compute_termination(self, 
                            state_dict: Dict[str,torch.Tensor], 
                            act_batch: Optional[torch.Tensor]=None,
                            compute_full_state: bool = False):

        if compute_full_state:
            state_dict = self.compute_full_state(state_dict)
        
        info = {}
        q_pos_batch = state_dict['q_pos']
        q_vel_batch = state_dict['q_vel']
        q_acc_batch = state_dict['q_acc']
        orig_size = q_pos_batch.size()[0:-1]

        termination = torch.zeros(orig_size, device=self.device).flatten()
        term_cost = torch.zeros(orig_size, device=self.device).flatten()

        if self.cfg['cost']['primitive_collision']['weight'] > 0:
            with record_function('arm_task:primitive_collision_cost'):
                # link_pos_batch, link_rot_batch = state_dict['link_pos'], state_dict['link_rot']
                # link_pos_batch = link_pos_batch.view(-1, self.num_links, 3)
                # link_rot_batch = link_rot_batch.view(-1, self.num_links, 3, 3)
                # link_pose_dict = state_dict['link_pose_dict']
                # print("entered primitive collision cost")
                link_spheres_dict = state_dict['link_spheres_dict']
                self_coll_dist = state_dict['self_coll_dist']
                # coll_cost, coll_cost_info = self.primitive_collision_cost.forward(link_pos_batch, link_rot_batch)
                coll_cost, coll_cost_info = self.primitive_collision_cost.forward(link_spheres_dict, self_coll_dist)
                collision_violation = torch.logical_or(coll_cost_info['in_coll_world'], coll_cost_info['in_coll_self'])
                collision_violation = collision_violation.sum(-1)
                termination = torch.logical_or(termination, collision_violation)
                term_cost += coll_cost
                info = coll_cost_info
        if self.cfg['cost']['state_bound']['weight'] > 0:
            with record_function('arm_task:bound_cost'):
                # print("entered bound cost")
                state_batch = torch.cat((q_pos_batch, q_vel_batch, q_acc_batch), dim=-1).view(-1, 3*self.n_dofs)
                bound_cost, bound_cost_info = self.bound_cost.forward(state_batch)
                in_bounds = bound_cost_info['in_bounds']#[..., 0:2*self.n_dofs] #only use qpos and qvel
                bounds_violation = torch.logical_not(in_bounds).sum(-1)
                termination = torch.logical_or(termination, bounds_violation)
                term_cost += bound_cost
                info['in_bounds'] = bound_cost_info['in_bounds'].view(*orig_size, -1)
                info['bound_dist'] = bound_cost_info['bound_dist'].view(*orig_size, -1)

            with record_function('arm_task:compute_success'):
                success = self.compute_success(state_dict).flatten()
                termination = torch.logical_or(termination, success)
                info['success'] = success
            
        # termination = torch.logical_or(collision_violation, bounds_violation)
        # termination_cost = coll_cost +  bound_cost

        # if self.cfg['cost']['robot_self_collision']['weight'] > 0:
        #     #coll_cost = self.robot_self_collision_cost.forward(link_pos_batch, link_rot_batch)
        #     with record_function('self_collision_cost'):
        #         # coll_cost = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs])
        #         self_coll_cost = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs], link_pos_seq=link_pos_batch, link_rot_seq=link_rot_batch)
        #         self_coll_cost = self_coll_cost.view(self.num_instances, self.batch_size, self.horizon)
        #         termination += self_coll_cost > 0.

        termination = termination.float().view(orig_size)
        term_cost = term_cost.view(orig_size)

        return termination, term_cost, info

    @torch.jit.export
    def compute_full_state(self, state_dict: Dict[str,torch.Tensor], debug:bool=False)->Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]:
        # if 'state_seq' not in state_dict:
        # print('in compute full state')
        # for k in state_dict:
        #     print(k, state_dict[k].device)

        q_pos = state_dict['q_pos'].to(device=self.device)
        q_vel = state_dict['q_vel'].to(device=self.device)
        q_acc = state_dict['q_acc'].to(device=self.device)

        # ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch, ee_lin_vel, ee_ang_vel = self.robot_model.compute_fk_and_jacobian(
        #     q_pos.reshape(-1, q_pos.shape[-1]), q_vel.view(-1, self.n_dofs), self.cfg['ee_link_name'])
        st=time.time()
        #check time + device
        with record_function("arm_task: fk + jacobian"):
            link_pose_dict, link_spheres_dict, self_coll_dist, lin_jac, ang_jac = self.robot_model.compute_fk_and_jacobian(
                q_pos.view(-1, self.n_dofs), q_vel.view(-1, self.n_dofs),
                link_name=self.ee_link_name) #could be issue
        # print('compute_full_state:fk', time.time()-st)
        
        ee_pos, ee_rot = link_pose_dict[self.ee_link_name]
        st1 = time.time()
        ee_quat = matrix_to_quaternion(ee_rot) #Could be issue
        # print("compute_full_state:quat", time.time()-st1)

        # ee_pos = ee_pos.view(*orig_size, -1)
        # ee_rot = ee_rot.view(*orig_size, 3, 3)
        # ee_quat = ee_quat.view(*orig_size, -1)
        # lin_jac = lin_jac.view(*orig_size, 3, -1)
        # ang_jac = ang_jac.view(*orig_size, 3, -1)
        st3 = time.time()
        ee_pos = ee_pos.view(*q_pos.shape[0:-1], -1)
        ee_rot = ee_rot.view(*q_pos.shape[0:-1], 3, 3)
        ee_quat = ee_quat.view(*q_pos.shape[0:-1], -1)
        lin_jac = lin_jac.view(*q_pos.shape[0:-1], 3, -1)
        ang_jac = ang_jac.view(*q_pos.shape[0:-1], 3, -1)

        ee_jac = torch.cat((ang_jac, lin_jac), dim=-2) #check
        ee_vel_twist = torch.matmul(ee_jac, q_vel.unsqueeze(-1)).squeeze(-1)
        ee_acc_twist = torch.matmul(ee_jac, q_acc.unsqueeze(-1)).squeeze(-1)
        # print("compute_full_state:reshape", time.time()-st3)

        # get link poses:
        # link_pos_list = []
        # link_rot_list = []
        # for ki,k in enumerate(self.link_names):
        #     # link_pos, link_rot = self.robot_model.get_link_pose(k)
        #     curr_link_pos, curr_link_rot = link_pose_dict[k]
        #     # link_pos_seq[:,:,:,ki,:] = link_pos.view((self.num_instances, self.batch_size, self.horizon, 3))
        #     # link_rot_seq[:,:,:,ki,:,:] = link_rot.view((self.num_instances, self.batch_size, self.horizon, 3,3))
        #     # link_pos_seq[:,ki,:] = link_pos.view((self.num_instances, 1, 3))
        #     # link_rot_seq[:,ki,:,:] = link_rot.view((self.num_instances, 1, 3,3))
        #     link_pos_list.append(curr_link_pos.unsqueeze(1))
        #     link_rot_list.append(curr_link_rot.unsqueeze(1))
        
        # link_pos = torch.cat(link_pos_list, dim=2)
        # link_rot = torch.cat(link_rot_list, dim=2)

        new_state_dict = {}
        for k in state_dict.keys():
            new_state_dict[k] = state_dict[k]
        
        new_state_dict['ee_pos'] =  ee_pos 
        new_state_dict['ee_rot'] = ee_rot
        new_state_dict['ee_quat'] = ee_quat
        new_state_dict['ee_jacobian'] = ee_jac
        new_state_dict['ee_vel_twist'] = ee_vel_twist
        new_state_dict['ee_acc_twist'] = ee_acc_twist
        new_state_dict['link_pose_dict'] = link_pose_dict
        new_state_dict['link_spheres_dict'] = link_spheres_dict
        new_state_dict['self_coll_dist'] = self_coll_dist

        # new_state_dict['link_pos'] = link_pos
        # new_state_dict['link_rot'] = link_rot

        # self.prev_state_buff = self.prev_state_buff.roll(-1, dims=1)
        # self.prev_state_buff[:,-1,:] = new_state_dict['state_seq'].clone()
        # new_state_dict['prev_state_seq'] = self.prev_state_buff
        return new_state_dict
        
    
    def compute_success(self, state_dict:Dict[str,torch.Tensor]):
        q_pos = state_dict['q_pos']
        return torch.zeros(q_pos.shape[0:-1], device=self.device)        

    def compute_metrics(self):
        pass

    def reset(self, rng:Optional[torch.Generator]=None):
        pass

    def reset_idx(self, rng:Optional[torch.Generator]=None):
        pass

    def update_params(self, retract_state):
        pass

    def init_viewer(self):
        pass
        # if not self.vis_initialized:
        #     print('Initializing rollout viewer')
        #     import meshcat
        #     import meshcat.geometry as meshcat_g

        #     self.vis = meshcat.Visualizer() #if self.vis is None else self.vis
        #     self.vis_initialized = True

        #     for h in range(self.horizon):
        #         robot_material = meshcat_g.MeshBasicMaterial(
        #             color=0xff0000, transparent=True, opacity=1.0 / (1.0 + 0.1*h))
        #         goal_material = meshcat_g.MeshBasicMaterial(
        #             color=0x00FF00, transparent=True, opacity=1.0 / (1.0 + 0.1*h))
                
        #         self.vis["world"]["robot"][str(h)].set_object(meshcat_g.Sphere(self.robot_radius), robot_material)
        #         self.vis["world"]["goal"][str(h)].set_object(meshcat_g.Sphere(self.robot_radius), goal_material)

        #     self.fig, self.ax = plt.subplots(self.action_dim)
        
    
    def visualize_rollouts(self, rollout_data):
        pass
        # self.init_viewer()
        #     # self.fig.canvas.restore_region(self.bg)
        # import meshcat.transformations as meshcat_tf

        # robot_pos = rollout_data['states']['q_pos']
        # costs = rollout_data['costs']
        # _, _, horizon = costs.shape
        # gamma_seq = torch.cumprod(torch.tensor([1.0] + [0.99] * (horizon - 1)),dim=0).reshape(1, horizon)
        # gamma_seq = gamma_seq.to(self.device)
        # total_costs = cost_to_go(costs, gamma_seq)[:, :, 0]
        # top_values, top_idx = torch.topk(total_costs, 10, dim=-1)
        # top_idx = top_idx.squeeze(0)

        # top_robot_pos = torch.index_select(robot_pos, 1, top_idx).squeeze(0).cpu() #.squeeze(0)
        # top_robot_pos = torch.cat((top_robot_pos, torch.zeros(10, horizon, 1)), dim=-1)

        # robot_goal = self.robot_goal_buff.clone().cpu()
        # robot_goal = torch.cat((robot_goal, torch.zeros(self.num_instances, 1)), dim=-1).numpy()

        # for i in range(horizon):
        #     self.vis["world"]["robot"][str(i)].set_transform(meshcat_tf.translation_matrix(top_robot_pos[0,i]))
        #     self.vis["world"]["goal"][str(i)].set_transform(meshcat_tf.translation_matrix(robot_goal[0]))
        
        # #Pliot the actions as well
        # actions = rollout_data['actions'].cpu().numpy()
        # _, b, h, nd = actions.shape 
        #     # fig, ax = plt.subplots(nd)

        # for d_i in range(nd):
        #     self.ax[d_i].clear()
        #     for b_i in range(b):
        #         data = actions[0, b_i, :, d_i]
        #         self.ax[d_i].plot(data)
        # plt.pause(0.01)
        # plt.draw()

    @property
    def obs_dim(self)->int:
        # return 18 #+ 2*self.n_dofs 
        return 24 #3 #12 #9 #21 #24

    @property
    def action_dim(self)->int:
        return self.n_dofs

    @property
    def state_bounds(self)->torch.Tensor:
        return self.bounds

    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)
        