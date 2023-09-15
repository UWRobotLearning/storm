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
from typing import Optional, Dict
import torch
from torch.profiler import record_function
import time

from ..cost import DistCost, PoseCost, ProjectedDistCost, JacobianCost, ZeroCost, EEVelCost, StopCost, FiniteDifferenceCost
from ..cost.bound_cost import BoundCost
from ..cost.manipulability_cost import ManipulabilityCost
from ..cost import CollisionCost, VoxelCollisionCost, PrimitiveCollisionCost
from ..model import URDFKinematicModel
from ...util_file import join_path, get_assets_path
from ...differentiable_robot_model.coordinate_transform import matrix_to_quaternion, quaternion_to_matrix
from ...mpc.model.integration_utils import build_fd_matrix
from ...mpc.rollout.rollout_base import RolloutBase
from ..cost.robot_self_collision_cost import RobotSelfCollisionCost

class ArmBase(RolloutBase):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update cfg to be kwargs
    """

    def __init__(self, cfg, world_params=None, value_function=None, viz_rollouts=False, device=torch.device('cpu'), dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.cfg = cfg
        self.viz_rollouts = viz_rollouts
        # mppi_params = cfg['mppi']
        model_params = cfg['model']
        robot_params = model_params # cfg['robot_params']
        self.batch_size = cfg['batch_size']
        self.horizon = cfg['horizon']
        self.num_instances = cfg['num_instances']
        
        assets_path = get_assets_path()
        # initialize dynamics model:
        # dynamics_horizon = cfg['horizon'] * model_params['dt']
        #Create the dynamical system used for rollouts
        self.dynamics_model = URDFKinematicModel(
            join_path(assets_path, cfg['model']['urdf_path']),
            batch_size=cfg['batch_size'],
            # horizon=dynamics_horizon,
            horizon=cfg['horizon'],
            num_instances=self.num_instances,
            ee_link_name=cfg['model']['ee_link_name'],
            link_names=cfg['model']['link_names'],
            dt_traj_params=cfg['model']['dt_traj_params'],
            # max_acc=cfg['model']['max_acc'],
            control_space=cfg['control_space'],
            # vel_scale=cfg['model']['vel_scale'],
            device=self.device) #dtype=self.dtype

        # self.dt = self.dynamics_model.dt
        self.n_dofs = self.dynamics_model.n_dofs
        # rollout traj_dt starts from dt->dt*(horizon+1) as tstep 0 is the current state
        #self.traj_dt = torch.arange(self.dt, (mppi_params['horizon'] + 1) * self.dt, self.dt, device=device, dtype=float_dtype)
        self.traj_dt = self.dynamics_model.traj_dt
        self.num_links = len(cfg['model']['link_names'])
                        
        # self.jacobian_cost = JacobianCost(ndofs=self.n_dofs, device=device,
        #                                   float_dtype=float_dtype,
        #                                   retract_weight=cfg['cost']['retract_weight'])
        
        self.null_cost = ProjectedDistCost(ndofs=self.n_dofs, device=device, float_dtype=dtype,
                                           **cfg['cost']['null_space'])
        
        self.manipulability_cost = ManipulabilityCost(ndofs=self.n_dofs, device=device,
                                                      float_dtype=dtype,
                                                      **cfg['cost']['manipulability'])

        self.zero_vel_cost = ZeroCost(device=device, float_dtype=dtype, **cfg['cost']['zero_vel'])

        self.zero_acc_cost = ZeroCost(device=device, float_dtype=dtype, **cfg['cost']['zero_acc'])

        tensor_args = {'device': self.device, 'dtype': self.dtype}
        
        self.stop_cost = StopCost(**cfg['cost']['stop_cost'],
                                  tensor_args=tensor_args,
                                  traj_dt=self.traj_dt)
        
        self.stop_cost_acc = StopCost(**cfg['cost']['stop_cost_acc'],
                                      tensor_args=tensor_args,
                                      traj_dt=self.traj_dt)

        self.retract_state = torch.tensor([self.cfg['cost']['retract_state']], device=device, dtype=dtype)

        # if 'smooth' in self.cfg['cost']:
        #     self.fd_matrix = build_fd_matrix(10 - self.cfg['cost']['smooth']['order'], device=self.device, dtype=self.dtype, PREV_STATE=True, order=self.cfg['cost']['smooth']['order'])

        # if self.cfg['cost']['smooth']['weight'] > 0:
        #     self.smooth_cost = FiniteDifferenceCost(**self.cfg['cost']['smooth'],
        #                                             tensor_args=tensor_args)

        self.primitive_collision_cost = PrimitiveCollisionCost(
            world_params=world_params, robot_params=robot_params, 
            batch_size=self.num_instances * self.batch_size * self.horizon,
            device=self.device, **self.cfg['cost']['primitive_collision'])

        # if cfg['cost']['robot_self_collision']['weight'] > 0.0:
        # self.robot_self_collision_cost = RobotSelfCollisionCost(
        #     config=model_params['robot_collision_params'], batch_size=self.num_instances * self.batch_size * self.horizon,
        #     device=self.device, **self.cfg['cost']['robot_self_collision'])

        self.ee_vel_cost = EEVelCost(ndofs=self.n_dofs,device=device, float_dtype=dtype,**cfg['cost']['ee_vel'])

        bounds = torch.cat([self.dynamics_model.state_lower_bounds[:2*self.n_dofs].unsqueeze(0), 
                            self.dynamics_model.state_upper_bounds[:2*self.n_dofs].unsqueeze(0)], dim=0).T
        
        self.bound_cost = BoundCost(**cfg['cost']['state_bound'],
                                    tensor_args=tensor_args,
                                    bounds=bounds)

        self.link_pos_seq = torch.zeros((self.num_instances, self.num_links, 3), dtype=self.dtype, device=self.device)
        self.link_rot_seq = torch.zeros((self.num_instances, self.num_links, 3, 3), dtype=self.dtype, device=self.device)
    
    def compute_cost(
            self, 
            state_dict: Dict[str, torch.Tensor], 
            action_batch: Optional[torch.Tensor]=None,
            termination: Optional[Dict[str, torch.Tensor]]=None, 
            # no_coll:bool=False, 
            horizon_cost:bool=True):

        cost_terms = {}

        # if state_dict is None:
        state_dict = self.compute_full_state(state_dict)

        # num_instances, curr_batch_size, num_traj_points, _ = state_dict['state_seq'].shape
        state_batch = state_dict['state_seq'].view(self.num_instances * self.batch_size, self.horizon, -1)
        lin_jac_batch, ang_jac_batch = state_dict['lin_jac_seq'], state_dict['ang_jac_seq']
        lin_jac_batch = lin_jac_batch.view(self.num_instances*self.batch_size, self.horizon, 3, self.n_dofs)
        ang_jac_batch = ang_jac_batch.view(self.num_instances*self.batch_size, self.horizon, 3, self.n_dofs)

        link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        link_pos_batch = link_pos_batch.view(self.num_instances*self.batch_size, self.horizon, self.num_links, 3)
        link_rot_batch = link_rot_batch.view(self.num_instances*self.batch_size, self.horizon, self.num_links, 3, 3)
        # prev_state = state_dict['prev_state_seq']
        # prev_state = prev_state.view(self.num_instances*prev_state.shape[-2], prev_state.shape[-1])
        # prev_state_tstep = state_dict['prev_state_seq'][:,-1]

        retract_state = self.retract_state
                
        J_full = torch.cat((lin_jac_batch, ang_jac_batch), dim=-2)
        

        #null-space cost
        # if self.cfg['cost']['null_space']['weight'] > 0:
        null_disp_cost, _ = self.null_cost.forward(
            state_batch[:,:,0:self.n_dofs] -
            retract_state[:,0:self.n_dofs],
            J_full,
            proj_type='identity',
            dist_type='squared_l2')
        
        cost = null_disp_cost
        
        if self.cfg['cost']['manipulability']['weight'] > 0.0:
            with record_function('manipulability_cost'):
                cost += self.manipulability_cost.forward(J_full)

        if self.cfg['cost']['state_bound']['weight'] > 0:
            with record_function('bound_cost'):
                cost += self.bound_cost.forward(state_batch[:,:,:2*self.n_dofs])

        if self.cfg['cost']['ee_vel']['weight'] > 0:
            with record_function('ee_vel_cost'):
                cost += self.ee_vel_cost.forward(state_batch, lin_jac_batch)

        # if no_coll and (not horizon_cost):
        #     return cost, state_dict

        if horizon_cost:
            if self.cfg['cost']['stop_cost']['weight'] > 0:
                with record_function("stop_cost"):
                    cost += self.stop_cost.forward(state_batch[:, :, self.n_dofs:2*self.n_dofs])

            if self.cfg['cost']['stop_cost_acc']['weight'] > 0:
                with record_function("stop_cost_acc"):
                    cost += self.stop_cost_acc.forward(state_batch[:, 2*self.n_dofs :3*self.n_dofs])


        if termination is not None:
            termination = termination.view(self.num_instances*self.batch_size, self.horizon)
            termination_cost = 5000.0 * termination 
            cost += termination_cost
            cost_terms['termination'] = termination_cost


            # if self.cfg['cost']['smooth']['weight'] > 0:
            #     with record_function('smooth_cost'):
            #         order = self.cfg['cost']['smooth']['order']
            #         prev_dt = (self.fd_matrix @ prev_state_tstep)[-order:]
            #         n_mul = 1
            #         state = state_batch[:,:, self.n_dofs * n_mul:self.n_dofs * (n_mul+1)]
            #         p_state = prev_state[-order:,self.n_dofs * n_mul: self.n_dofs * (n_mul+1)].unsqueeze(0)
            #         p_state = p_state.expand(state.shape[0], -1, -1)
            #         state_buffer = torch.cat((p_state, state), dim=1)
            #         traj_dt = torch.cat((prev_dt, self.traj_dt))
            #         cost += self.smooth_cost.forward(state_buffer, traj_dt)

        # # if not no_coll:
        # if self.cfg['cost']['robot_self_collision']['weight'] > 0:
        #     with record_function('self_collision_cost'):
        #         st = time.time()
        #         coll_cost = self.robot_self_collision_cost.forward(
        #             state_batch[:,:,:self.n_dofs], 
        #             link_pos_seq=link_pos_batch, link_rot_seq=link_rot_batch)
        #         cost += coll_cost
        #         print('self', time.time()-st)

        # if self.cfg['cost']['primitive_collision']['weight'] > 0:


        #     if self.cfg['cost']['voxel_collision']['weight'] > 0:
        #         with record_function('voxel_collision'):
        #             coll_cost = self.voxel_collision_cost.forward(link_pos_batch, link_rot_batch)
        #             cost += coll_cost

        cost = cost.view(self.num_instances, self.batch_size, self.horizon)

        return cost, cost_terms, state_dict

    def compute_observations(self, 
                             state_dict: Dict[str,torch.Tensor]):
        
        
        # ee_quat_seq =  matrix_to_quaternion(state_dict['ee_rot_seq'])
        # obs = torch.cat(
        #     (state_dict['state_seq'][:,:,:,0:2*self.n_dofs],
        #     state_dict['ee_pos_seq'],
        #     state_dict['ee_rot_seq'].flatten(-2,-1)),
        #     dim=-1
        # )
        state_dict = self.compute_full_state(state_dict)
        # return obs
        obs = torch.cat(
            (state_dict['q_pos'], state_dict['q_vel']), dim=-1
        )
        return obs, state_dict


    def compute_termination(self, state_dict: Dict[str,torch.Tensor], act_batch: torch.Tensor):
        

        state_dict = self.compute_full_state(state_dict)

        # num_instances, curr_batch_size, num_traj_points, _ = state_dict['state_seq'].shape
        termination = torch.zeros(self.num_instances, self.batch_size, self.horizon, device=self.device)

        state_batch = state_dict['state_seq'].view(self.num_instances * self.batch_size, self.horizon, -1)
        lin_jac_batch, ang_jac_batch = state_dict['lin_jac_seq'], state_dict['ang_jac_seq']
        lin_jac_batch = lin_jac_batch.view(self.num_instances*self.batch_size, self.horizon, 3, self.n_dofs)
        ang_jac_batch = ang_jac_batch.view(self.num_instances*self.batch_size, self.horizon, 3, self.n_dofs)

        link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        link_pos_batch = link_pos_batch.view(self.num_instances*self.batch_size, self.horizon, self.num_links, 3)
        link_rot_batch = link_rot_batch.view(self.num_instances*self.batch_size, self.horizon, self.num_links, 3, 3)

        with record_function('primitive_collision'):
            coll_cost = self.primitive_collision_cost.forward(link_pos_batch, link_rot_batch)
            termination += coll_cost > 0

        # if self.cfg['cost']['state_bound']['weight'] > 0:
        #     with record_function('bound_cost'):
        #         bound_cost = self.bound_cost.forward(
        #             state_batch[:,:,:2*self.n_dofs]).view(self.num_instances, self.batch_size, self.horizon)
        #         termination += bound_cost > 0.
        #         print(bound_cost)
        #         print(termination)

        # if self.cfg['cost']['robot_self_collision']['weight'] > 0:
        #     #coll_cost = self.robot_self_collision_cost.forward(link_pos_batch, link_rot_batch)
        #     with record_function('self_collision_cost'):
        #         # coll_cost = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs])
        #         self_coll_cost = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs], link_pos_seq=link_pos_batch, link_rot_seq=link_rot_batch)
        #         self_coll_cost = self_coll_cost.view(self.num_instances, self.batch_size, self.horizon)
        #         termination += self_coll_cost > 0.

        
        # if self.cfg['cost']['primitive_collision']['weight'] > 0:
        #     with record_function('primitive_collision'):
        #         env_coll_cost = self.primitive_collision_cost.forward(link_pos_batch, link_rot_batch)
        #         env_coll_cost = env_coll_cost.view(self.num_instances, self.batch_size, self.horizon)
        #         termination += env_coll_cost > 0.

        termination = (termination > 0)
        
        return termination, state_dict
    
    def rollout_fn(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Parameters
        ----------
        action_seq: torch.Tensor [num_particles, horizon, d_act]
        """

        with record_function("robot_model"):
            state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        
        #link_pos_seq, link_rot_seq = self.dynamics_model.get_link_poses()
        with record_function("compute_termination"):
            term_seq, _ = self.compute_termination(state_dict, act_seq)

        with record_function("compute_cost"):
            cost_seq, _, _ = self.compute_cost(state_dict, act_seq, termination=term_seq)

        sim_trajs = dict(
            actions=act_seq,#.clone(),
            costs=cost_seq,#clone(),
            terminations=term_seq,
            ee_pos_seq=state_dict['ee_pos_seq'],#.clone(),
            value_preds=None,
            #link_pos_seq=link_pos_seq,
            #link_rot_seq=link_rot_seq,
            rollout_time=0.0
        )
        
        return sim_trajs

    def compute_full_state(self, state_dict: Dict[str,torch.Tensor]):

        if 'state_seq' not in state_dict:
            q_pos = state_dict['q_pos'].to(device=self.device)
            q_vel = state_dict['q_vel'].to(device=self.device)
            q_acc = state_dict['q_acc'].to(device=self.device)
            tstep = state_dict['tstep']

            current_state_tensor = torch.cat((q_pos, q_vel, q_acc, tstep), dim=-1)
            
            # num_instances = current_state_tensor.shape[0]
            # num_traj_points = 1 
            # horizon = 1 #self.dynamics_model.num_traj_points
            
            # ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(
            #     current_state[:, :self.dynamics_model.n_dofs], 
            #     current_state[:, self.dynamics_model.n_dofs: 2*self.dynamics_model.n_dofs], 
            #     self.cfg['model']['ee_link_name'])
            ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(
                q_pos, q_vel, self.cfg['model']['ee_link_name'])

            link_pos_seq = self.link_pos_seq
            link_rot_seq = self.link_rot_seq

            # get link poses:
            for ki,k in enumerate(self.dynamics_model.link_names):
                link_pos, link_rot = self.dynamics_model.robot_model.get_link_pose(k)
                # link_pos_seq[:,:,:,ki,:] = link_pos.view((self.num_instances, self.batch_size, self.horizon, 3))
                # link_rot_seq[:,:,:,ki,:,:] = link_rot.view((self.num_instances, self.batch_size, self.horizon, 3,3))
                link_pos_seq[:,ki,:] = link_pos.view((self.num_instances, 1, 3))
                link_rot_seq[:,ki,:,:] = link_rot.view((self.num_instances, 1, 3,3))
                
            # if len(current_state_tensor.shape) == 2:
            #     current_state_tensor = current_state_tensor.unsqueeze(1).unsqueeze(1)
            #     ee_pos_batch = ee_pos_batch.unsqueeze(1).unsqueeze(1)
            #     ee_rot_batch = ee_rot_batch.unsqueeze(1).unsqueeze(1)
            #     lin_jac_batch = lin_jac_batch.unsqueeze(1).unsqueeze(1)
            #     ang_jac_batch = ang_jac_batch.unsqueeze(1).unsqueeze(1)

            new_state_dict = {}

            for k in state_dict.keys():
                new_state_dict[k] = state_dict[k].clone()
            new_state_dict['state_seq'] = current_state_tensor
            new_state_dict['ee_pos_seq'] =  ee_pos_batch 
            new_state_dict['ee_rot_seq'] = ee_rot_batch
            new_state_dict['lin_jac_seq'] = lin_jac_batch 
            new_state_dict['ang_jac_seq'] =  ang_jac_batch
            new_state_dict['link_pos_seq'] = link_pos_seq 
            new_state_dict['link_rot_seq'] = link_rot_seq
            # 'prev_state_seq': current_state_tensor
            return new_state_dict
        
        return state_dict


    def update_params(self, retract_state=None):
        """
        Updates the goal targets for the cost functions.

        """
        
        if retract_state is not None:
            self.retract_state = torch.as_tensor(retract_state, device=self.device, dtype=self.dtype).unsqueeze(0)
        
        return True


    # def get_ee_pose(self, current_state: torch.Tensor):
    #     current_state = current_state.to(device=self.device, dtype=self.dtype)
         
        
    #     ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(
    #         current_state[:,:self.dynamics_model.n_dofs], current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], self.cfg['model']['ee_link_name'])

    #     ee_quat = matrix_to_quaternion(ee_rot_batch)
    #     state = {'ee_pos_seq': ee_pos_batch, 'ee_rot_seq': ee_rot_batch,
    #              'lin_jac_seq': lin_jac_batch, 'ang_jac_seq': ang_jac_batch,
    #              'ee_quat_seq': ee_quat}
    #     return state

    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)
        