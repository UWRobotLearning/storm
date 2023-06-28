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

    # def __init__(self, cfg, tensor_args={'device':"cpu", 'dtype':torch.float32}, world_params=None):
    def __init__(self, cfg, world_params=None, device=torch.device('cpu'), dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.cfg = cfg
        # mppi_params = cfg['mppi']
        model_params = cfg['model']
        robot_params = model_params # cfg['robot_params']
        self.num_instances = cfg['num_instances']
        
        assets_path = get_assets_path()
        # initialize dynamics model:
        dynamics_horizon = cfg['horizon'] * model_params['dt']
        #Create the dynamical system used for rollouts
        # self.dynamics_model = torch.jit.script(URDFKinematicModel(join_path(assets_path,cfg['model']['urdf_path']),
        #                                          dt=cfg['model']['dt'],
        #                                          batch_size=mppi_params['num_particles'],
        #                                          horizon=dynamics_horizon,
        #                                          num_instances=self.num_instances,
        #                                         #  tensor_args=self.tensor_args,
        #                                          ee_link_name=cfg['model']['ee_link_name'],
        #                                          link_names=cfg['model']['link_names'],
        #                                          dt_traj_params=cfg['model']['dt_traj_params'],
        #                                          control_space=cfg['control_space'],
        #                                          vel_scale=cfg['model']['vel_scale'],
        #                                          device=self.device,
        #                                          dtype=self.dtype))

        self.dynamics_model = torch.jit.script(URDFKinematicModel(join_path(assets_path,cfg['model']['urdf_path']),
                                                 dt=cfg['model']['dt'],
                                                 batch_size=cfg['num_particles'],
                                                 horizon=dynamics_horizon,
                                                 num_instances=self.num_instances,
                                                #  tensor_args=self.tensor_args,
                                                 ee_link_name=cfg['model']['ee_link_name'],
                                                 link_names=cfg['model']['link_names'],
                                                 dt_traj_params=cfg['model']['dt_traj_params'],
                                                 control_space=cfg['control_space'],
                                                 vel_scale=cfg['model']['vel_scale'],
                                                 device=self.device,
                                                 dtype=self.dtype))

        self.dt = self.dynamics_model.dt
        self.n_dofs = self.dynamics_model.n_dofs
        # rollout traj_dt starts from dt->dt*(horizon+1) as tstep 0 is the current state
        #self.traj_dt = torch.arange(self.dt, (mppi_params['horizon'] + 1) * self.dt, self.dt, device=device, dtype=float_dtype)
        self.traj_dt = self.dynamics_model.traj_dt
        self.num_links = len(cfg['model']['link_names'])
                
        self.goal_state = None
        self.goal_ee_pos = None
        self.goal_ee_rot = None
        
        # device = self.tensor_args['device']
        # float_dtype = self.tensor_args['dtype']

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

        if 'smooth' in self.cfg['cost']:
            self.fd_matrix = build_fd_matrix(10 - self.cfg['cost']['smooth']['order'], device=self.device, dtype=self.dtype, PREV_STATE=True, order=self.cfg['cost']['smooth']['order'])

        if self.cfg['cost']['smooth']['weight'] > 0:
            self.smooth_cost = FiniteDifferenceCost(**self.cfg['cost']['smooth'],
                                                    tensor_args=tensor_args)

        if self.cfg['cost']['voxel_collision']['weight'] > 0:
            self.voxel_collision_cost = VoxelCollisionCost(robot_params=robot_params,
                                                           tensor_args=tensor_args,
                                                           **self.cfg['cost']['voxel_collision'])
            
        if cfg['cost']['primitive_collision']['weight'] > 0.0:
            self.primitive_collision_cost = PrimitiveCollisionCost(world_params=world_params, robot_params=robot_params, tensor_args=tensor_args, **self.cfg['cost']['primitive_collision'])

        if cfg['cost']['robot_self_collision']['weight'] > 0.0:
            self.robot_self_collision_cost = RobotSelfCollisionCost(config=model_params['robot_collision_params'], device=self.device, **self.cfg['cost']['robot_self_collision'])

        self.ee_vel_cost = EEVelCost(ndofs=self.n_dofs,device=device, float_dtype=dtype,**cfg['cost']['ee_vel'])

        bounds = torch.cat([self.dynamics_model.state_lower_bounds[:self.n_dofs * 3].unsqueeze(0),self.dynamics_model.state_upper_bounds[:self.n_dofs * 3].unsqueeze(0)], dim=0).T
        self.bound_cost = BoundCost(**cfg['cost']['state_bound'],
                                    tensor_args=tensor_args,
                                    bounds=bounds)

        self.link_pos_seq = torch.zeros((self.num_instances, 1, 1, self.num_links, 3), dtype=self.dtype, device=self.device)
        self.link_rot_seq = torch.zeros((self.num_instances, 1, 1, self.num_links, 3, 3), dtype=self.dtype, device=self.device)
    
    def compute_cost(self, state_dict=None, action_batch=None, current_state=None, no_coll:bool=False, horizon_cost:bool=True):

        if current_state is None and state_dict is None:
            raise ValueError('Either provide current state or full robot state dict')
        
        if state_dict is None:
            state_dict = self.compute_full_robot_state(current_state)

        num_instances, curr_batch_size, num_traj_points, _ = state_dict['state_seq'].shape
        
        state_batch = state_dict['state_seq'].view(num_instances * curr_batch_size, num_traj_points, -1)
        lin_jac_batch, ang_jac_batch = state_dict['lin_jac_seq'], state_dict['ang_jac_seq']
        lin_jac_batch = lin_jac_batch.view(num_instances*curr_batch_size, num_traj_points, 3, self.n_dofs)
        ang_jac_batch = ang_jac_batch.view(num_instances*curr_batch_size, num_traj_points, 3, self.n_dofs)

        link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        link_pos_batch = link_pos_batch.view(num_instances*curr_batch_size, num_traj_points, self.num_links, 3)
        link_rot_batch = link_rot_batch.view(num_instances*curr_batch_size, num_traj_points, self.num_links, 3, 3)
        prev_state = state_dict['prev_state_seq']
        prev_state = prev_state.view(num_instances*prev_state.shape[-2], prev_state.shape[-1])
        prev_state_tstep = state_dict['prev_state_seq'][:,-1]

        retract_state = self.retract_state
                
        J_full = torch.cat((lin_jac_batch, ang_jac_batch), dim=-2)
        

        #null-space cost
        # if self.cfg['cost']['null_space']['weight'] > 0:
        null_disp_cost = self.null_cost.forward(state_batch[:,:,0:self.n_dofs] -
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
                cost += self.bound_cost.forward(state_batch[:,:,:self.n_dofs * 3])

        if self.cfg['cost']['ee_vel']['weight'] > 0:
            with record_function('ee_vel_cost'):
                cost += self.ee_vel_cost.forward(state_batch, lin_jac_batch)

        if no_coll and (not horizon_cost):
            return cost, state_dict

        if horizon_cost:
            if self.cfg['cost']['stop_cost']['weight'] > 0:
                with record_function("stop_cost"):
                    cost += self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])

            if self.cfg['cost']['stop_cost_acc']['weight'] > 0:
                with record_function("stop_cost_acc"):
                    cost += self.stop_cost_acc.forward(state_batch[:, self.n_dofs*2 :self.n_dofs * 3])

            if self.cfg['cost']['smooth']['weight'] > 0:
                with record_function('smooth_cost'):
                    order = self.cfg['cost']['smooth']['order']
                    prev_dt = (self.fd_matrix @ prev_state_tstep)[-order:]
                    n_mul = 1
                    state = state_batch[:,:, self.n_dofs * n_mul:self.n_dofs * (n_mul+1)]
                    p_state = prev_state[-order:,self.n_dofs * n_mul: self.n_dofs * (n_mul+1)].unsqueeze(0)
                    p_state = p_state.expand(state.shape[0], -1, -1)
                    state_buffer = torch.cat((p_state, state), dim=1)
                    traj_dt = torch.cat((prev_dt, self.traj_dt))
                    cost += self.smooth_cost.forward(state_buffer, traj_dt)

        if not no_coll:
            if self.cfg['cost']['robot_self_collision']['weight'] > 0:
                #coll_cost = self.robot_self_collision_cost.forward(link_pos_batch, link_rot_batch)
                with record_function('self_collision_cost'):
                    coll_cost = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs])
                    cost += coll_cost
            if self.cfg['cost']['primitive_collision']['weight'] > 0:
                with record_function('primitive_collision'):
                    coll_cost = self.primitive_collision_cost.forward(link_pos_batch, link_rot_batch)
                    cost += coll_cost
            if self.cfg['cost']['voxel_collision']['weight'] > 0:
                with record_function('voxel_collision'):
                    coll_cost = self.voxel_collision_cost.forward(link_pos_batch, link_rot_batch)
                    cost += coll_cost

        return cost, state_dict

    def compute_observations(self, state_dict=None, current_state=None):
        if current_state is None and state_dict is None:
            raise ValueError('Either provide current state or full robot state dict')
        
        if state_dict is None:
            state_dict = self.compute_full_robot_state(current_state)
        
        # ee_quat_seq =  matrix_to_quaternion(state_dict['ee_rot_seq'])
        obs = torch.cat(
            (state_dict['state_seq'][:,:,:,0:2*self.n_dofs],
            state_dict['ee_pos_seq'],
            state_dict['ee_rot_seq'].flatten(-2,-1)),
            dim=-1
        )
        return obs, state_dict

    def compute_termination(self, state_dict, act_batch):
        return super().compute_termination(state_dict, act_batch)
    
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
        with record_function("cost_fns"):
            cost_seq, _ = self.compute_cost(state_dict, act_seq)

        sim_trajs = dict(
            actions=act_seq,#.clone(),
            costs=cost_seq,#clone(),
            ee_pos_seq=state_dict['ee_pos_seq'],#.clone(),
            #link_pos_seq=link_pos_seq,
            #link_rot_seq=link_rot_seq,
            rollout_time=0.0
        )
        
        return sim_trajs

 
    # def compute_current_cost(self, current_state:torch.Tensor=None, action_batch:torch.Tensor=None, state_dict= None, no_coll:bool=True):    
    #     if current_state is None and state_dict is None:
    #         raise ValueError('Either provide current state or full robot state dict')
    #     if state_dict is None:
    #         state_dict = self.compute_full_robot_state(current_state)
    #     cost = self.compute_cost(state_dict, action_batch, no_coll=no_coll, horizon_cost=False, return_dist=True)
    #     return cost, state_dict

    # def compute_current_observation(self, current_state:torch.Tensor=None, action_batch:torch.Tensor=None, state_dict= None, no_coll:bool=True):



    def compute_full_robot_state(self, current_state: torch.Tensor):

        current_state = current_state.to(device=self.device, dtype=self.dtype)
        
        curr_batch_size = current_state.shape[0]
        num_traj_points = 1 
        horizon = 1 #self.dynamics_model.num_traj_points
        
        ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(
            current_state[:, :self.dynamics_model.n_dofs], 
            current_state[:, self.dynamics_model.n_dofs: 2*self.dynamics_model.n_dofs], 
            self.cfg['model']['ee_link_name'])

        link_pos_seq = self.link_pos_seq
        link_rot_seq = self.link_rot_seq

        # get link poses:
        for ki,k in enumerate(self.dynamics_model.link_names):
            link_pos, link_rot = self.dynamics_model.robot_model.get_link_pose(k)
            link_pos_seq[:,:,:,ki,:] = link_pos.view((curr_batch_size, horizon, num_traj_points,3))
            link_rot_seq[:,:,:,ki,:,:] = link_rot.view((curr_batch_size, horizon, num_traj_points,3,3))
            
        if len(current_state.shape) == 2:
            current_state = current_state.unsqueeze(1).unsqueeze(1)
            ee_pos_batch = ee_pos_batch.unsqueeze(1).unsqueeze(1)
            ee_rot_batch = ee_rot_batch.unsqueeze(1).unsqueeze(1)
            lin_jac_batch = lin_jac_batch.unsqueeze(1).unsqueeze(1)
            ang_jac_batch = ang_jac_batch.unsqueeze(1).unsqueeze(1)
        
        state_dict = {'ee_pos_seq': ee_pos_batch, 'ee_rot_seq':ee_rot_batch,
                      'lin_jac_seq': lin_jac_batch, 'ang_jac_seq': ang_jac_batch,
                      'state_seq': current_state, 'prev_state_seq': current_state,
                      'link_pos_seq':link_pos_seq, 'link_rot_seq':link_rot_seq}
        
        return state_dict


    def update_params(self, retract_state=None):
        """
        Updates the goal targets for the cost functions.

        """
        
        if retract_state is not None:
            self.retract_state = torch.as_tensor(retract_state, device=self.device, dtype=self.dtype).unsqueeze(0)
        
        return True


    def get_ee_pose(self, current_state):
        current_state = current_state.to(device=self.device, dtype=self.dtype)
         
        
        ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(current_state[:,:self.dynamics_model.n_dofs], current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], self.cfg['model']['ee_link_name'])

        ee_quat = matrix_to_quaternion(ee_rot_batch)
        state = {'ee_pos_seq':ee_pos_batch, 'ee_rot_seq':ee_rot_batch,
                 'lin_jac_seq': lin_jac_batch, 'ang_jac_seq': ang_jac_batch,
                 'ee_quat_seq':ee_quat}
        return state

    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)