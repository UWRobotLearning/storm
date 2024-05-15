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

from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torch.profiler import record_function

from ...differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.spatial_vector_algebra import matrix_to_quaternion
from .integration_utils import build_int_matrix, build_fd_matrix, tensor_step_acc, tensor_step_vel, tensor_step_pos, tensor_step_jerk
from storm_kit.learning.policies import Policy

class URDFKinematicModel(nn.Module):
    link_names: List[str]
    robot_keys: List[str]
    joint_lim_dicts: List[Dict[str, float]]
    _integrate_matrix: torch.Tensor
    _fd_matrix: torch.Tensor
    _fd_matrix_jerk: torch.Tensor
    dt_traj_params: Dict[str, float]
    state_seq: torch.Tensor
    prev_state_buffer: torch.Tensor

    def __init__(
        self, 
        cfg,
        device: torch.device = torch.device('cpu')):
        
        super().__init__()
        # self.urdf_path:str = urdf_path
        # self.robot_config = robot_config
        self.cfg = cfg
        self.robot_keys:List[str] = cfg['robot_keys']
        self.device:torch.device = device
        self.dtype = torch.float32
        # self.dt = dt
        self.robot_cfg = cfg['robot']
        self.ee_link_name:str = cfg['robot']['ee_link_name']
        self.state_batch_size:int = 1
        self.batch_size:int = cfg['batch_size']
        self.horizon:int = cfg['horizon'] #horizon
        self.num_traj_points:int = self.horizon 

        self.robot_model = torch.jit.script(DifferentiableRobotModel(cfg.robot, device=self.device))
        #self.robot_model.half()
        self.n_dofs:int = self.robot_model._n_dofs
        self.d_state:int = 3 * self.n_dofs + 1
        self.d_action:int = self.n_dofs
        # self.max_acc = max_acc
        # self.max_jerk = max_jerk

        #Variables for enforcing joint limits
        self.joint_lim_dicts:List[Dict[str, float]] = self.robot_model.get_joint_limits()
        self.state_upper_bounds:torch.Tensor = torch.zeros(2*self.n_dofs, device=self.device)
        self.state_lower_bounds:torch.Tensor = torch.zeros(2*self.n_dofs, device=self.device)
        for i in range(self.n_dofs):
            self.state_upper_bounds[i] = self.joint_lim_dicts[i]['upper']
            self.state_lower_bounds[i] = self.joint_lim_dicts[i]['lower']
            self.state_upper_bounds[i+self.n_dofs] = self.joint_lim_dicts[i]['velocity'] #* vel_scale
            self.state_lower_bounds[i+self.n_dofs] = -1.0 * self.joint_lim_dicts[i]['velocity'] #* vel_scale
            # self.state_upper_bounds[i+2*self.n_dofs] = max_acc
            # self.state_lower_bounds[i+2*self.n_dofs] = -1.0 * max_acc
        
        # # #pre-allocating memory for rollouts
        self.Z:torch.Tensor = torch.tensor([0.], device=self.device) #torch.zeros(batch_size, self.n_dofs, device=self.device)
        self._integrate_matrix:torch.Tensor = build_int_matrix(self.num_traj_points, device=self.device, dtype=self.dtype, diagonal=-1)
        # self._integrate_matrix = self._integrate_matrix.unsqueeze(0).repeat(self.num_instances, 1, 1)
        self.control_space:str = self.cfg['control_space']
        
        if self.control_space == 'acc':
            self.step_fn = tensor_step_acc
        elif self.control_space == 'vel':
            self.step_fn = tensor_step_vel
        elif self.control_space == 'jerk':
            self.step_fn = tensor_step_jerk
        elif self.control_space == 'pos':
            self.step_fn = tensor_step_pos

        self._fd_matrix:torch.Tensor = build_fd_matrix(self.num_traj_points, device=self.device, order=1)
        self._fd_matrix_jerk:torch.Tensor = build_fd_matrix(self.num_traj_points+1, device=self.device, order=1)

        self.dt_traj_params = self.cfg['dt_traj_params']
        # if dt_traj_params is None or self.num_traj_points <= 1:
        #     dt_array = [self.dt] * int(1.0 * self.num_traj_points) 
        if self.num_traj_points <= 1:
            dt_array = [self.dt_traj_params['base_dt']] * int(1.0 * self.num_traj_points)
        else:
            dt_array = [self.dt_traj_params['base_dt']] * int(self.dt_traj_params['base_ratio'] * self.num_traj_points)
            smooth_blending = torch.linspace(self.dt_traj_params['base_dt'], self.dt_traj_params['max_dt'], steps=int((1 - self.dt_traj_params['base_ratio']) * self.num_traj_points)).tolist()
            dt_array += smooth_blending
            # self.dt = dt_traj_params['base_dt']
        if len(dt_array) != self.num_traj_points:
            dt_array.insert(0, dt_array[0])
        
        # self._dt_h = torch.tensor(dt_array, dtype=self.dtype, device=self.device)
        # self.dt_traj = self._dt_h
        # self.traj_dt = self._dt_h
        self.traj_dt = torch.tensor(dt_array, device=self.device) #dt between each step
        self.traj_tstep = torch.matmul(self._integrate_matrix, self.traj_dt) #cumulative time difference

        # self.link_pos_seq = torch.empty((self.num_instances, self.batch_size, self.num_traj_points, len(self.link_names),3), device=self.device)
        # self.link_rot_seq = torch.empty((self.num_instances, self.batch_size, self.num_traj_points, len(self.link_names),3,3), device=self.device)

        # self.prev_state_buffer = torch.zeros((self.num_instances, 10, self.d_state), device=self.device) 
        # self.prev_state_fd = build_fd_matrix(10, device=self.device, order=1)

        self.action_order:int = 0
        self._integrate_matrix_nth = build_int_matrix(self.num_traj_points, order=self.action_order, device=self.device, dtype=self.dtype, traj_dt=self.traj_dt)
        self._integrate_matrix_nth = self._integrate_matrix_nth.unsqueeze(0).repeat(self.state_batch_size, 1, 1).unsqueeze(1)
        # self._nth_traj_dt = torch.pow(self.traj_dt, self.action_order)
        self.initial_step = True

        self.allocate_buffers(self.state_batch_size, self.batch_size)


    def allocate_buffers(self, state_batch_size:int, batch_size:int):
        self.state_seq = torch.zeros(state_batch_size, batch_size, self.num_traj_points, self.d_state, device=self.device)
        self.prev_state_buffer = torch.zeros((state_batch_size, 10, self.d_state), device=self.device) 

    def forward(self, curr_state: torch.Tensor, act:torch.Tensor, dt) -> torch.Tensor: 
        return self.get_next_state(curr_state, act, dt)

    @torch.jit.export
    def get_next_state(
        self, curr_state: torch.Tensor, act:torch.Tensor, 
        dt:float)->torch.Tensor:
        """ Does a single step from the current state
        Args:
        curr_state: current state
        act: action
        dt: time to integrate
        Returns:
        next_state
        """

        if self.control_space == 'jerk':
            curr_state[..., 2 * self.n_dofs:3 * self.n_dofs] = curr_state[:, self.n_dofs:2*self.n_dofs] + act * dt
            curr_state[..., self.n_dofs:2*self.n_dofs] = curr_state[:, self.n_dofs:2*self.n_dofs] + curr_state[self.n_dofs*2:self.n_dofs*3] * dt
            curr_state[..., :self.n_dofs] = curr_state[:, :self.n_dofs] + curr_state[:, self.n_dofs:2*self.n_dofs] * dt
        elif self.control_space == 'acc':
            curr_state[..., 2 * self.n_dofs:3 * self.n_dofs] = act 
            curr_state[..., self.n_dofs:2*self.n_dofs] = curr_state[..., self.n_dofs:2*self.n_dofs] + curr_state[..., 2*self.n_dofs:3*self.n_dofs] * dt
            curr_state[..., :self.n_dofs] = curr_state[..., :self.n_dofs] + curr_state[..., self.n_dofs:2*self.n_dofs] * dt
        elif self.control_space == 'vel':
            curr_state[..., 2 * self.n_dofs:3 * self.n_dofs] = 0.0
            curr_state[..., self.n_dofs:2*self.n_dofs] = act #* dt
            curr_state[..., :self.n_dofs] = curr_state[:, :self.n_dofs] + curr_state[:, self.n_dofs:2*self.n_dofs] * dt
        elif self.control_space == 'pos':
            curr_state[..., 2 * self.n_dofs:3 * self.n_dofs] = 0.0
            curr_state[..., 1 * self.n_dofs:2 * self.n_dofs] = 0.0
            curr_state[..., :self.n_dofs] = act
        
        return curr_state
    
    @torch.jit.export
    def tensor_step(self, state: torch.Tensor, act: torch.Tensor, state_seq: torch.Tensor, batch_size:int, horizon:int) -> torch.Tensor:
        """
        Args:
        state: [1,N]
        act: [H,N]
        todo:
        Integration with variable dt along trajectory
        """
        inp_device = act.device
        state = state.to(self.device)
        act = act.to(self.device)
        # nth_act_seq = self.integrate_action(act)
        # state_seq = self.step_fn(state, nth_act_seq, state_seq, self.traj_dt, self._integrate_matrix, self._fd_matrix, self.n_dofs, self.num_instances, batch_size, horizon) #, self._fd_matrix)
        state_seq = self.step_fn(
            state, act, state_seq, self.traj_dt, self._integrate_matrix, 
            self._fd_matrix, self.n_dofs, self.state_batch_size,
            batch_size, horizon)
        #state_seq = self.enforce_bounds(state_seq)
        state_seq[..., -1] = self.traj_tstep # timestep array
        return state_seq.to(inp_device)
        
    @torch.jit.export
    def rollout_open_loop(self, start_state_dict: Dict[str, torch.Tensor], act_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        act_seq = act_seq.to(self.device)
        state_batch_size = act_seq.shape[0]
        batch_size = act_seq.shape[1]

        if batch_size != self.batch_size or state_batch_size != self.state_batch_size:
            self.batch_size = batch_size
            self.state_batch_size = state_batch_size
            self.allocate_buffers(self.state_batch_size, self.batch_size)

        start_q_pos = start_state_dict[self.robot_keys[0]]
        start_q_vel = start_state_dict[self.robot_keys[1]]
        start_q_acc = start_state_dict[self.robot_keys[2]]
        # import pdb; pdb.set_trace()
        start_object_pos = start_state_dict[self.robot_keys[3]]
        relative_object_pos = start_state_dict[self.robot_keys[4]]
        start_cube_pos = start_state_dict[self.robot_keys[5]]
        start_t = start_state_dict['tstep']

        curr_robot_state = torch.cat((start_q_pos, start_q_vel, start_q_acc, start_t), dim=-1)
        # curr_robot_state = curr_robot_state#.unsqueeze(1)

        self.prev_state_buffer = self.prev_state_buffer.roll(-1, dims=1)
        self.prev_state_buffer[:,-1,:] = curr_robot_state

        # # add start state to prev state buffer:
        # if self.initial_step:
            # self.prev_state_buffer = torch.zeros((self.num_instances, 10, self.d_state), device=self.device, dtype=self.dtype)
        # self.prev_state_buffer[:,:,:] = curr_robot_state.unsqueeze(1)
            # self.initial_step = False


        # compute dt w.r.t previous data?
        state_seq = self.state_seq
        # ee_pos_seq = self.ee_pos_seq
        # ee_rot_seq = self.ee_rot_seq
        # curr_horizon = self.horizon
        curr_batch_size = self.batch_size
        num_traj_points = self.num_traj_points
        # link_pos_seq = self.link_pos_seq
        # link_rot_seq = self.link_rot_seq
        #         
        # curr_state = self.prev_state_buffer[:,-1:,:self.n_dofs * 3]
        curr_state = curr_robot_state[..., 0:3*self.n_dofs]
        with record_function("tensor_step"):
            state_seq = self.tensor_step(curr_state, act_seq, state_seq, curr_batch_size, num_traj_points)

        # state_dict = self.compute_full_state(state_seq)

        state_dict:Dict[str, torch.Tensor] = {
            'q_pos': state_seq[..., :self.n_dofs],
            'q_vel': state_seq[..., self.n_dofs:2*self.n_dofs],
            'q_acc':  state_seq[..., 2*self.n_dofs:3*self.n_dofs],
            'start_object_pos': start_object_pos,
            'start_q_pos': start_q_pos,
            'relative_object_pos': relative_object_pos,
            'start_cube_pos': start_cube_pos,
            'tstep':  state_seq[..., -1]
        }
        return state_dict #self.compute_full_state(state_seq)


    # @torch.jit.export
    # def compute_rollouts(
    #     self, start_state_dict: Dict[str, torch.Tensor], 
    #     open_loop_act_seq: Optional[torch.Tensor]=None,
    #     sampling_policy: Optional[Policy]=None,
    #     num_policy_rollouts:int=0) -> Dict[str, torch.Tensor]:
        
    #     num_open_loop_rollouts = open_loop_act_seq.shape[1] if open_loop_act_seq is not None else 0
    #     # TODO: clamp actions?

    #     start_q_pos = start_state_dict[self.robot_keys[0]]
    #     start_q_vel = start_state_dict[self.robot_keys[1]]
    #     start_q_acc = start_state_dict[self.robot_keys[2]]
    #     start_t = start_state_dict['tstep']

    #     curr_robot_state = torch.cat((start_q_pos, start_q_vel, start_q_acc, start_t), dim=-1)
    #     # curr_robot_state = torch.cat((start_state_dict[k] for k in start_state_dict), dim=-1)
    #     state_batch_size = curr_robot_state.shape[0]
    #     batch_size = num_open_loop_rollouts + num_policy_rollouts
        
    #     if batch_size != self.batch_size or state_batch_size != self.state_batch_size:
    #         self.batch_size = batch_size
    #         self.state_batch_size = state_batch_size
    #         #TODO: Just call reset here?
    #         self.allocate_buffers(self.state_batch_size, self.batch_size)

    #     self.prev_state_buffer = self.prev_state_buffer.roll(-1, dims=1)
    #     self.prev_state_buffer[:,-1,:] = curr_robot_state

    #     # curr_robot_state = curr_robot_state.unsqueeze(1)
    #     # self.prev_state_buffer = self.prev_state_buffer.roll(-1, dims=1)
    #     # self.prev_state_buffer[:,-1,:] = curr_robot_state#.squeeze(1)

    #     state_seq = self.state_seq

    #     # curr_state = self.prev_state_buffer[:,-1:,:self.n_dofs * 3]
    #     curr_state = curr_robot_state[..., 0:3*self.n_dofs]
    #     if num_open_loop_rollouts > 0:
    #         with record_function("tensor_step"):
    #             # forward step with step matrix:
    #             state_seq[:,:num_open_loop_rollouts] = \
    #                 self.tensor_step(
    #                     curr_state, open_loop_act_seq, 
    #                     state_seq[:,:num_open_loop_rollouts], 
    #                     num_open_loop_rollouts, self.num_traj_points)

    #     info = {'cl_act_seq': None}
    #     if sampling_policy is not None and num_policy_rollouts > 0:
    #         with record_function("policy_rollout"):
    #             state_seq[:,num_open_loop_rollouts:], info = \
    #                 self.rollout_closed_loop(
    #                     curr_robot_state, sampling_policy, 
    #                     state_seq[:, num_open_loop_rollouts:], 
    #                     num_policy_rollouts, self.num_traj_points)


    #     state_dict = self.compute_full_state(state_seq)

    #     return state_dict, info

    def compute_full_state(self, state_seq:torch.Tensor)->Dict[str, torch.Tensor]:

        q_pos_seq = state_seq[..., 0:self.n_dofs]
        q_vel_seq = state_seq[..., self.n_dofs:2*self.n_dofs]
        q_acc_seq = state_seq[..., 2*self.n_dofs:3*self.n_dofs]
        tstep_seq = state_seq[..., -1]

        # ee_pos_seq = self.ee_pos_seq
        # ee_rot_seq = self.ee_rot_seq
        # curr_horizon = self.horizon
        num_traj_points = self.num_traj_points
        # link_pos_seq = self.link_pos_seq
        # link_rot_seq = self.link_rot_seq
        curr_state_batch_size = state_seq.shape[0]
        curr_batch_size = state_seq.shape[1]

        #Compute jerk sequence
        prev_acc = self.prev_state_buffer[:, -1:, 2*self.n_dofs:3*self.n_dofs].unsqueeze(1)
        # # prev_acc = self.prev_state_buffer[:,-1, 2*self.n_dofs:3*self.n_dofs].unsqueeze(0).unsqueeze(0).expand(q_acc_seq.shape[0],-1,-1)
        prev_acc = prev_acc.expand(curr_state_batch_size, curr_batch_size, -1, -1)
        q_jerk_seq = torch.cat((prev_acc, q_acc_seq), dim=-2)
        q_jerk_seq = torch.einsum('bijk, lj->bilk', q_jerk_seq, self._fd_matrix_jerk)
        # # q_jerk_seq = torch.einsum('bij, li->blj', q_jerk_seq, self._fd_matrix_jerk)
        q_jerk_seq = torch.div(q_jerk_seq, self.traj_dt[:, None])

        # with record_function("urdf_kinematic_model: fk + jacobian"):
        #     ee_pos_seq, ee_rot_seq, lin_jac_seq, ang_jac_seq, ee_lin_vel, ee_ang_vel = self.robot_model.compute_fk_and_jacobian(
        #         q_pos_seq.view(-1, self.n_dofs), q_vel_seq.view(-1, self.n_dofs),
        #         link_name=self.ee_link_name)
        with record_function("urdf_kinematic_model: fk + jacobian"):
            link_pose_dict, link_spheres_dict, self_coll_dist, lin_jac_seq, ang_jac_seq = self.robot_model.compute_fk_and_jacobian(
                q_pos_seq.view(-1, self.n_dofs), q_vel_seq.view(-1, self.n_dofs),
                link_name=self.ee_link_name)

        # get link poses:
        # for ki,k in enumerate(self.link_names):
        #     link_pos, link_rot = self.robot_model.get_link_pose(k)
        # for (k, (link_pos, link_rot)) in link_pos_dict.items():
        # for (ki, k) in enumerate(self.link_names):
        #     link_pos, link_rot = link_pose_dict[k]
        #     link_pos_seq[:,:,:,ki,:] = link_pos.view((curr_state_batch_size, curr_batch_size, num_traj_points, 3))
        #     link_rot_seq[:,:,:,ki,:,:] = link_rot.view((curr_state_batch_size, curr_batch_size, num_traj_points, 3, 3))            
            # link_pos_seq[...,ki,:] = link_pos.view((curr_batch_size, num_traj_points, 3))
            # link_rot_seq[...,ki,:,:] = link_rot.view((curr_batch_size, num_traj_points, 3, 3))

        # for link_name in self.link_names:
        #     print(link_pose_dict[link_name][0].shape)
            # link_pos  = link_pose_dict[link_name][0].view((curr_batch_size, num_traj_points, 3))
            # link_rot = link_pose_dict[link_name][1].view((curr_batch_size, num_traj_points, 3,3))
            # link_pose_dict[link_name] = (link_pos, link_rot)

        ee_pos_seq, ee_rot_seq = link_pose_dict[self.ee_link_name]
        ee_quat_seq = matrix_to_quaternion(ee_rot_seq)


        ee_pos_seq = ee_pos_seq.view((curr_state_batch_size, curr_batch_size, num_traj_points, 3))
        ee_rot_seq = ee_rot_seq.view((curr_state_batch_size, curr_batch_size, num_traj_points, 3, 3))
        ee_quat_seq = ee_quat_seq.view((curr_state_batch_size, curr_batch_size, num_traj_points, 4))
        lin_jac_seq = lin_jac_seq.view((curr_state_batch_size, curr_batch_size, num_traj_points, 3, self.n_dofs))
        ang_jac_seq = ang_jac_seq.view((curr_state_batch_size, curr_batch_size, num_traj_points, 3, self.n_dofs))
        # ee_pos_seq = ee_pos_seq.view((curr_batch_size, num_traj_points, 3))
        # ee_rot_seq = ee_rot_seq.view((curr_batch_size, num_traj_points, 3, 3))
        # lin_jac_seq = lin_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        # ang_jac_seq = ang_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        ee_jacobian_seq = torch.cat((ang_jac_seq, lin_jac_seq), dim=-2)

        ee_vel_twist_seq = torch.matmul(ee_jacobian_seq, q_vel_seq.unsqueeze(-1)).squeeze(-1)
        # this is a first order approximation of ee acceleation i.e ignoreing \dot{J}
        # TODO: Test this against finite differencing and/or RNE
        ee_acc_twist_seq = torch.matmul(ee_jacobian_seq, q_acc_seq.unsqueeze(-1)).squeeze(-1)

        state_dict = {
            # 'state_seq': state_seq, 
            'q_pos': q_pos_seq,
            'q_vel': q_vel_seq,
            'q_acc': q_acc_seq,
            'q_jerk': q_jerk_seq,
            'ee_pos': ee_pos_seq,
            'ee_rot': ee_rot_seq,
            'ee_quat': ee_quat_seq,
            'ee_jacobian': ee_jacobian_seq,
            'ee_vel_twist': ee_vel_twist_seq,
            'ee_acc_twist': ee_acc_twist_seq,
            # 'link_pos': link_pos_seq,
            # 'link_rot': link_rot_seq,
            'link_pose_dict': link_pose_dict,
            'link_spheres_dict': link_spheres_dict,
            'self_coll_dist': self_coll_dist,
            'prev_state': self.prev_state_buffer,
            'tstep': tstep_seq}
        
        return state_dict

    def enforce_bounds(self, state_batch: torch.Tensor)->torch.Tensor:
        """
            Project state into bounds
        """
        batch_size = state_batch.shape[0]
        bounded_state = torch.max(torch.min(state_batch, self.state_upper_bounds), self.state_lower_bounds)
        bounded_q = bounded_state[...,:,:self.n_dofs]
        bounded_qd = bounded_state[...,:,self.n_dofs:2*self.n_dofs]
        bounded_qdd = bounded_state[...,:,2*self.n_dofs:3*self.n_dofs]
        
        # #set velocity and acc to zero where position is at bound
        bounded_qd = torch.where(bounded_q < self.state_upper_bounds[:self.n_dofs], bounded_qd, self.Z)
        bounded_qd = torch.where(bounded_q > self.state_lower_bounds[:self.n_dofs], bounded_qd, self.Z)
        bounded_qdd = torch.where(bounded_q < self.state_upper_bounds[:self.n_dofs], bounded_qdd, -10.0*bounded_qdd)
        bounded_qdd = torch.where(bounded_q > self.state_lower_bounds[:self.n_dofs], bounded_qdd, -10.0*bounded_qdd)

        # #set acc to zero where vel is at bounds 
        bounded_qdd = torch.where(bounded_qd < self.state_upper_bounds[self.n_dofs:2*self.n_dofs], bounded_qdd, self.Z)
        bounded_qdd = torch.where(bounded_qd > self.state_lower_bounds[self.n_dofs:2*self.n_dofs], bounded_qdd, self.Z)
        state_batch[...,:,:self.n_dofs] = bounded_q
        #state_batch[...,:,self.n_dofs:self.n_dofs*2] = bounded_qd
        #state_batch[...,:,self.n_dofs*2:self.n_dofs*3] = bounded_qdd
        
        #bounded_state = torch.cat((bounded_q, bounded_qd, bounded_qdd), dim=-1) 
        return state_batch

    def integrate_action(self, act_seq: torch.Tensor) -> torch.Tensor:
        if self.action_order == 0:
            return act_seq
        nth_act_seq = self._integrate_matrix_nth  @ act_seq
        return nth_act_seq

    # def integrate_action_step(self, act:torch.Tensor, dt:float)->torch.Tensor:
    #     for _ in range(self.action_order):
    #         act = act * dt
    #     return act

    def reset(self, reset_data=None):
        # self.prev_state_buffer = torch.zeros((self.num_instances, 10, self.d_state), device=self.device, dtype=self.dtype) 
        # self.prev_state_buffer = torch.zeros((10, self.d_state), device=self.device, dtype=self.dtype) 
        self.initial_step = True
        self.allocate_buffers(self.state_batch_size, self.batch_size)




        # q_pos_seq = state_seq[..., 0:self.n_dofs]
        # q_vel_seq = state_seq[..., self.n_dofs:2*self.n_dofs]
        # q_acc_seq = state_seq[..., 2*self.n_dofs:3*self.n_dofs]
        # tstep_seq = state_seq[..., -1]

        # #Compute jerk sequence
        # prev_acc = self.prev_state_buffer[:,-1, 2*self.n_dofs:3*self.n_dofs].unsqueeze(1).unsqueeze(1)
        # prev_acc = prev_acc.expand(self.state_batch_size, self.batch_size, 1, -1)
        # q_jerk_seq = torch.cat((prev_acc, q_acc_seq), dim=-2)
        # q_jerk_seq = torch.einsum('bijk, lj->bilk', q_jerk_seq, self._fd_matrix_jerk)
        # q_jerk_seq = torch.div(q_jerk_seq, self.traj_dt[:, None])

        # shape_tup = (self.state_batch_size * curr_batch_size * num_traj_points, self.n_dofs)
        # with record_function("fk + jacobian"):
        #     ee_pos_seq, ee_rot_seq, lin_jac_seq, ang_jac_seq = self.robot_model.compute_fk_and_jacobian(
        #         state_seq[:,:,:,:self.n_dofs].view(shape_tup),
        #         link_name=self.ee_link_name)

        # # get link poses:
        # for ki,k in enumerate(self.link_names):
        #     link_pos, link_rot = self.robot_model.get_link_pose(k)
        #     link_pos_seq[:,:,:,ki,:] = link_pos.view((self.state_batch_size, curr_batch_size, num_traj_points, 3))
        #     link_rot_seq[:,:,:,ki,:,:] = link_rot.view((self.state_batch_size, curr_batch_size, num_traj_points, 3, 3))
            
        
        # ee_pos_seq = ee_pos_seq.view((self.state_batch_size, curr_batch_size, num_traj_points, 3))
        # ee_rot_seq = ee_rot_seq.view((self.state_batch_size, curr_batch_size, num_traj_points, 3, 3))
        # lin_jac_seq = lin_jac_seq.view((self.state_batch_size, curr_batch_size, num_traj_points, 3, self.n_dofs))
        # ang_jac_seq = ang_jac_seq.view((self.state_batch_size, curr_batch_size, num_traj_points, 3, self.n_dofs))
        # ee_jacobian_seq = torch.cat((ang_jac_seq, lin_jac_seq), dim=-2)

        # ee_vel_twist_seq = torch.matmul(ee_jacobian_seq, q_vel_seq.unsqueeze(-1)).squeeze(-1)
        # # this is a first order approximation of ee acceleation i.e ignoreing \dot{J}
        # # TODO: Test this against finite differencing and/or RNE
        # ee_acc_twist_seq = torch.matmul(ee_jacobian_seq, q_acc_seq.unsqueeze(-1)).squeeze(-1) 

        # state_dict = {'state_seq': state_seq.to(inp_device),
        #               'q_pos_seq': q_pos_seq.to(inp_device),
        #               'q_vel_seq': q_vel_seq.to(inp_device),
        #               'q_acc_seq': q_acc_seq.to(inp_device),
        #               'q_jerk_seq': q_jerk_seq.to(inp_device),
        #               'ee_pos_seq': ee_pos_seq.to(inp_device),
        #               'ee_rot_seq': ee_rot_seq.to(inp_device),
        #               'ee_jacobian_seq': ee_jacobian_seq.to(inp_device),
        #               'ee_vel_twist_seq': ee_vel_twist_seq.to(inp_device),
        #               'ee_acc_twist_seq': ee_acc_twist_seq.to(inp_device),
        #               'link_pos_seq': link_pos_seq.to(inp_device),
        #               'link_rot_seq': link_rot_seq.to(inp_device),
        #               'prev_state_seq': self.prev_state_buffer.to(inp_device),
        #               'tstep_seq': tstep_seq.to(inp_device)}

        # return state_dict
            

    # def reset_idx(self, env_ids):
    #     self.prev_state_buffer[env_ids] = torch.zeros_like(self.prev_state_buffer[env_ids])


    # #Rendering
    # def render(self, state):
    #     from urdfpy import URDF
    #     self.urdfpy_robot = URDF.load(self.urdf_path) #only for visualization

    #     q = state[:, 0:self.n_dofs]
    #     state_dict = {}
    #     for (i,joint) in enumerate(self.urdfpy_robot.actuated_joints):
    #         state_dict[joint.name] = q[:,i].item()
    #     self.urdfpy_robot.show(cfg=state_dict,use_collision=True) 


    # def render_trajectory(self, state_list):
    #     state_dict = {}
    #     q = state_list[0][:, 0:self.n_dofs]
    #     for (i,joint) in enumerate(self.urdfpy_robot.actuated_joints):
    #         state_dict[joint.name] = [q[:,i].item()]
    #     for state in state_list[1:]:
    #         q = state[:, 0:self.n_dofs]
    #         for (i,joint) in enumerate(self.urdfpy_robot.actuated_joints):
    #             state_dict[joint.name].append(q[:,i].item())
    #     self.urdfpy_robot.animate(cfg_trajectory=state_dict,use_collision=True) 

