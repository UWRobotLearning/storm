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
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
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
from storm_kit.mpc.cost import CollisionCost, VoxelCollisionCost, PrimitiveCollisionCost
from storm_kit.mpc.cost.robot_self_collision_cost import RobotSelfCollisionCost
from storm_kit.mpc.model.integration_utils import build_fd_matrix

class ArmTask(nn.Module):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update cfg to be kwargs
    """

    def __init__(self, cfg, world_params=None, viz_rollouts:bool=False, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.viz_rollouts = viz_rollouts
        self.batch_size = cfg.get('batch_size', 1)
        self.horizon = cfg.get('horizon', 1)
        self.num_instances = cfg.get('num_instances', 1)
        self.dt_traj_params = cfg.get('dt_traj_params', None)

        robot_urdf = cfg.get('robot_urdf', None)
        if robot_urdf is None:
            robot_urdf = cfg['model'].get('urdf_path', None)

        urdf_path = join_path(get_assets_path(), robot_urdf)
        self.robot_model = DifferentiableRobotModel(urdf_path, None, device=self.device) #, dtype=self.dtype)
        self.n_dofs = self.robot_model._n_dofs
        self.link_names = cfg.get('robot_link_names')
        if self.link_names is None:
            self.link_names = cfg['model'].get('link_names')
        self.num_links = len(self.link_names)
                        
        # self.jacobian_cost = JacobianCost(ndofs=self.n_dofs, device=device,
        #                                   float_dtype=float_dtype,
        #                                   retract_weight=cfg['cost']['retract_weight'])
        
        # self.null_cost = ProjectedDistCost(ndofs=self.n_dofs, device=device, 
        #                                    **cfg['cost']['null_space'])
        
        self.manipulability_cost = ManipulabilityCost(device=self.device,
                                                      **cfg['cost']['manipulability'])
        
        self.zero_q_vel_cost = NormCost(**self.cfg['cost']['zero_q_vel'], device=self.device)
        self.zero_q_acc_cost = NormCost(**self.cfg['cost']['zero_q_acc'], device=self.device)
        
        if self.dt_traj_params is not None:
            if self.cfg['cost']['stop_cost']['weight'] > 0:
                self.stop_cost = StopCost(**cfg['cost']['stop_cost'],
                                            horizon=self.horizon,
                                            device=self.device,
                                            dt_traj_params=self.dt_traj_params)
            
            if self.cfg['cost']['stop_cost_acc']['weight'] > 0:        
                self.stop_cost_acc = StopCost(**cfg['cost']['stop_cost_acc'],
                                            horizon=self.horizon,
                                            device=self.device,
                                            dt_traj_params=self.dt_traj_params)


        self.ee_vel_twist_cost = NormCost(**self.cfg['cost']['ee_vel_twist'], device=self.device)
        self.ee_acc_twist_cost = NormCost(**self.cfg['cost']['ee_acc_twist'], device=self.device)
        self.retract_state = torch.tensor([self.cfg['cost']['retract_state']], device=device)


        if self.cfg['cost']['smooth_cost']['weight'] > 0:
            self.fd_matrix = build_fd_matrix(10 - self.cfg['cost']['smooth_cost']['order'] + 1, device=self.device, order=self.cfg['cost']['smooth_cost']['order'])
            self.smooth_cost = FiniteDifferenceCost(
                **self.cfg['cost']['smooth_cost'], 
                horizon = self.horizon + 1,
                device=self.device)

        self.primitive_collision_cost = PrimitiveCollisionCost(
            world_params=world_params, robot_collision_params=cfg.robot_collision_params,
            world_collision_params=cfg.world_collision_params, 
            batch_size=self.num_instances * self.batch_size * self.horizon,
            device=self.device, **self.cfg['cost']['primitive_collision'])

        # if cfg['cost']['robot_self_collision']['weight'] > 0.0:
        # self.robot_self_collision_cost = RobotSelfCollisionCost(
        #     config=model_params['robot_collision_params'], batch_size=self.num_instances * self.batch_size * self.horizon,
        #     device=self.device, **self.cfg['cost']['robot_self_collision'])

        #Initialize bound cost
        self.joint_lim_dicts = self.robot_model.get_joint_limits()
        self.state_upper_bounds = torch.zeros(2*self.n_dofs, device=self.device)
        self.state_lower_bounds = torch.zeros(2*self.n_dofs, device=self.device)
        for i in range(self.n_dofs):
            self.state_upper_bounds[i] = self.joint_lim_dicts[i]['upper']
            self.state_lower_bounds[i] = self.joint_lim_dicts[i]['lower']
            self.state_upper_bounds[i+self.n_dofs] = self.joint_lim_dicts[i]['velocity'] #* vel_scale
            self.state_lower_bounds[i+self.n_dofs] = -1.0 * self.joint_lim_dicts[i]['velocity'] #* vel_scale


        bounds = torch.cat([self.state_lower_bounds[:2*self.n_dofs].unsqueeze(0), 
                            self.state_upper_bounds[:2*self.n_dofs].unsqueeze(0)], dim=0).T
        
        self.bound_cost = BoundCost(**cfg['cost']['state_bound'],
                                    device=self.device,
                                    bounds=bounds)

        # self.link_pos_seq = torch.zeros((self.num_instances, self.num_links, 3), device=self.device)
        # self.link_rot_seq = torch.zeros((self.num_instances, self.num_links, 3, 3), device=self.device)

        self.vis_initialized = False
        self.full_state_dict = None
    
    def forward(self, state_dict: Dict[str, torch.Tensor], act_batch:Optional[torch.Tensor]=None):
        state_dict = self._compute_full_state(state_dict)
        obs = self.compute_observations(state_dict)
        termination, termination_cost = self.compute_termination(state_dict, act_batch)
        cost, cost_terms = self.compute_cost(state_dict, act_batch)
        if termination_cost is not None:
            cost += termination_cost
            cost_terms['termination'] = termination_cost

        return obs, cost, termination, cost_terms, termination_cost
        

    def compute_cost(
            self, 
            state_dict: Dict[str, torch.Tensor], 
            action_batch: Optional[torch.Tensor]=None,
            # termination_cost: Optional[Dict[str, torch.Tensor]]=None, 
            horizon_cost:bool=True):

        cost_terms = {}
        # state_dict = self.compute_full_state(state_dict)

        q_pos_batch = state_dict['q_pos_seq']
        orig_size = q_pos_batch.size()[0:-1]
        new_size = reduce(mul, list(orig_size))  

        # q_pos_batch = state_dict['q_pos_seq'].view(self.num_instances * self.batch_size, self.horizon, -1)
        # q_vel_batch = state_dict['q_vel_seq'].view(self.num_instances * self.batch_size, self.horizon, -1)
        # q_acc_batch = state_dict['q_acc_seq'].view(self.num_instances * self.batch_size, self.horizon, -1)
        q_pos_batch = state_dict['q_pos_seq'].view(new_size, -1)
        q_vel_batch = state_dict['q_vel_seq'].view(new_size, -1)
        q_acc_batch = state_dict['q_acc_seq'].view(new_size, -1)


        # action_batch = action_batch.view(self.num_instances * self.batch_size, self.horizon, -1)
        # ee_jacobian = state_dict['ee_jacobian_seq'].view(self.num_instances*self.batch_size, self.horizon, 6, -1)
        # ee_vel_twist_batch = state_dict['ee_vel_twist_seq'].view(self.num_instances*self.batch_size, self.horizon, -1)
        # ee_acc_twist_batch = state_dict['ee_acc_twist_seq'].view(self.num_instances*self.batch_size, self.horizon, -1)
        ee_jacobian = state_dict['ee_jacobian_seq'].view(new_size, 6, -1)
        ee_vel_twist_batch = state_dict['ee_vel_twist_seq'].view(new_size, -1)
        ee_acc_twist_batch = state_dict['ee_acc_twist_seq'].view(new_size, -1)

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
            cost = self.manipulability_cost.forward(ee_jacobian)
        
        if self.cfg['cost']['zero_q_vel']['weight'] > 0:
            with record_function('zero_q_vel_cost'):
                cost += self.zero_q_vel_cost.forward(q_vel_batch) #.view(self.num_instances * self.batch_size, self.horizon)

        if self.cfg['cost']['zero_q_acc']['weight'] > 0:
            with record_function('zero_q_acc_cost'):
                cost += self.zero_q_vel_cost.forward(q_acc_batch)#.view(self.num_instances * self.batch_size, self.horizon)

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
                    q_vel_batch = q_vel_batch.view(self.num_instances * self.batch_size, self.horizon, -1)
                    cost += self.stop_cost.forward(q_vel_batch).view(new_size)

            if self.cfg['cost']['stop_cost_acc']['weight'] > 0:
                with record_function("stop_cost_acc"):
                    q_acc_batch = q_acc_batch.view(self.num_instances * self.batch_size, self.horizon, -1)
                    cost += self.stop_cost_acc.forward(q_acc_batch).view(new_size)


        # with record_function('bound_cost'):
        #     bound_cost = self.bound_cost.forward(state_batch[:,:,:2*self.n_dofs])
        #     cost += bound_cost

        # if termination is not None:
        #     termination = termination.view(self.num_instances*self.batch_size, self.horizon)            
        #     termination_cost = 100.0 * termination 
        #     cost += termination_cost
        #     cost_terms['termination'] = termination_cost

        # # if not no_coll:
        # if self.cfg['cost']['robot_self_collision']['weight'] > 0:
        #     with record_function('self_collision_cost'):
        #         st = time.time()
        #         coll_cost = self.robot_self_collision_cost.forward(
        #             state_batch[:,:,:self.n_dofs], 
        #             link_pos_seq=link_pos_batch, link_rot_seq=link_rot_batch)
        #         cost += coll_cost
        #         print('self', time.time()-st)
        cost = cost.view(orig_size) #.view(self.num_instances, self.batch_size, self.horizon)

        return cost, cost_terms #, state_dict

    def compute_observations(self, 
                             state_dict: Dict[str,torch.Tensor]):
        
        q_pos_batch = state_dict['q_pos_seq']
        q_vel_batch = state_dict['q_vel_seq']
        # orig_size = q_pos_batch.size()[0:-1]
        # new_size = reduce(mul, list(orig_size))  

        # ee_vel_twist = state_dict['ee_vel_twist_seq']
        # ee_pos_seq = state_dict['ee_pos_seq']
        # ee_rot_seq = state_dict['ee_rot_seq']
        # if 'ee_quat_seq' not in state_dict:
        #     ee_quat_seq = matrix_to_quaternion(ee_rot_seq.view(self.num_instances*self.batch_size*self.horizon, 3, 3))
        #     ee_quat_seq = ee_quat_seq.view(self.num_instances, self.batch_size, self.horizon, -1)
        # else:
        #     ee_quat_seq = state_dict['ee_quat_seq']
        # obs = q_pos_batch
        obs = torch.cat((q_pos_batch, q_vel_batch), dim=-1)
        # obs = obs.view(*orig_size, -1)

        # obs = q_pos_batch.view(*orig_size, -1)
        # obs = torch.cat(
        #     (state_dict['q_pos_seq']), dim=-1) #, ee_pos_seq, ee_quat_seq, ee_vel_twist), dim=-1)
        return obs #, state_dict


    def compute_termination(self, state_dict: Dict[str,torch.Tensor], act_batch: Optional[torch.Tensor]=None):

        # state_dict = self.compute_full_state(state_dict)

        # num_instances, curr_batch_size, num_traj_points, _ = state_dict['state_seq'].shape
        # termination = torch.zeros(self.num_instances, self.batch_size, self.horizon, device=self.device)
        q_pos_batch = state_dict['q_pos_seq']
        orig_size = q_pos_batch.size()[0:-1]
        new_size = reduce(mul, list(orig_size))  

        state_batch = state_dict['state_seq'].view(new_size, -1)
        link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        link_pos_batch = link_pos_batch.view(new_size, self.num_links, 3)
        link_rot_batch = link_rot_batch.view(new_size, self.num_links, 3, 3)

        with record_function('primitive_collision'):
            coll_cost = self.primitive_collision_cost.forward(link_pos_batch, link_rot_batch)
            termination = coll_cost > 0
            termination_cost = coll_cost

        with record_function('bound_cost'):
            bound_cost = self.bound_cost.forward(state_batch[..., :2*self.n_dofs])
            termination += bound_cost > 0
            termination_cost += bound_cost

        # if self.cfg['cost']['robot_self_collision']['weight'] > 0:
        #     #coll_cost = self.robot_self_collision_cost.forward(link_pos_batch, link_rot_batch)
        #     with record_function('self_collision_cost'):
        #         # coll_cost = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs])
        #         self_coll_cost = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs], link_pos_seq=link_pos_batch, link_rot_seq=link_rot_batch)
        #         self_coll_cost = self_coll_cost.view(self.num_instances, self.batch_size, self.horizon)
        #         termination += self_coll_cost > 0.

        termination = (termination > 0).float()

        # num_term_states_per_traj = torch.count_nonzero(termination, dim=-1)

        #mark any trajectory with terminal states as terminal
        # termination[num_term_states_per_traj > 0] = 1.0
        # non_zero =  torch.nonzero(termination, as_tuple=False).long()
        # if non_zero.numel() > 0:
        #     # termination[non_zero[:,0], non_zero[:,1]:] = 1.0
        #     print(torch.count_nonzero(termination, dim=-1))
        termination_cost = termination_cost.view(orig_size)
        
        return termination, termination_cost
    
    # def update_state(self, state_dict: Dict[str, torch.Tensor]):
    #     self.full_state_dict = self.compute_full_state(state_dict)



    def _compute_full_state(self, state_dict: Dict[str,torch.Tensor], debug=False):
        # print('computing full state', debug)
        if 'state_seq' not in state_dict:
            q_pos = state_dict['q_pos'].to(device=self.device)
            q_vel = state_dict['q_vel'].to(device=self.device)
            q_acc = state_dict['q_acc'].to(device=self.device)
            tstep = state_dict['tstep']
            # if debug:
            #     q_pos_before = q_pos.clone()
            #     print('before', q_pos)

            current_state_tensor = torch.cat((q_pos, q_vel, q_acc, tstep), dim=-1)
            # num_instances = current_state_tensor.shape[0]
            # num_traj_points = 1 
            # horizon = 1 #self.dynamics_model.num_traj_points
            
            # ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(
            #     current_state[:, :self.dynamics_model.n_dofs], 
            #     current_state[:, self.dynamics_model.n_dofs: 2*self.dynamics_model.n_dofs], 
            #     self.cfg['model']['ee_link_name'])
            ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.robot_model.compute_fk_and_jacobian(
                q_pos, self.cfg['ee_link_name'])
        
            # if debug:
                # q_pos_after = q_pos.clone()
                # print('after', q_pos_after)
                # assert torch.allclose(q_pos_before, q_pos_after)

            
            ee_quat_batch = matrix_to_quaternion(ee_rot_batch)
            ee_jac_batch = torch.cat((ang_jac_batch, lin_jac_batch), dim=-2)
            ee_vel_twist_batch = torch.matmul(ee_jac_batch, q_vel.unsqueeze(-1)).squeeze(-1)
            ee_acc_twist_batch = torch.matmul(ee_jac_batch, q_acc.unsqueeze(-1)).squeeze(-1)

 
            # get link poses:
            link_pos_list = []
            link_rot_list = []
            for ki,k in enumerate(self.link_names):
                link_pos, link_rot = self.robot_model.get_link_pose(k)
                # link_pos_seq[:,:,:,ki,:] = link_pos.view((self.num_instances, self.batch_size, self.horizon, 3))
                # link_rot_seq[:,:,:,ki,:,:] = link_rot.view((self.num_instances, self.batch_size, self.horizon, 3,3))
                # link_pos_seq[:,ki,:] = link_pos.view((self.num_instances, 1, 3))
                # link_rot_seq[:,ki,:,:] = link_rot.view((self.num_instances, 1, 3,3))
                link_pos_list.append(link_pos.unsqueeze(1))
                link_rot_list.append(link_rot.unsqueeze(1))
            
            link_pos_seq = torch.cat(link_pos_list, dim=1)
            link_rot_seq = torch.cat(link_rot_list, dim=1)

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
            new_state_dict['q_pos_seq'] = q_pos.clone()
            new_state_dict['q_vel_seq'] = q_vel.clone()
            new_state_dict['q_acc_seq'] = q_acc.clone()
            new_state_dict['ee_pos_seq'] =  ee_pos_batch.clone() 
            new_state_dict['ee_rot_seq'] = ee_rot_batch.clone()
            new_state_dict['ee_quat_seq'] = ee_quat_batch.clone()
            new_state_dict['ee_jacobian_seq'] = ee_jac_batch.clone()
            new_state_dict['ee_vel_twist_seq'] = ee_vel_twist_batch.clone()
            new_state_dict['ee_acc_twist_seq'] = ee_acc_twist_batch.clone()
            new_state_dict['link_pos_seq'] = link_pos_seq.clone()
            new_state_dict['link_rot_seq'] = link_rot_seq.clone()

            # self.prev_state_buff = self.prev_state_buff.roll(-1, dims=1)
            # self.prev_state_buff[:,-1,:] = new_state_dict['state_seq'].clone()
            # new_state_dict['prev_state_seq'] = self.prev_state_buff


            return new_state_dict
        
        return state_dict

    def compute_metrics(self):
        pass

    def reset(self):
        pass

    def reset_idx(self):
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
        return 2*self.n_dofs

    @property
    def action_dim(self)->int:
        return self.n_dofs




    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)
        