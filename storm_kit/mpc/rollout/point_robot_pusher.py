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
from torch.profiler import record_function
import time

from ..cost import DistCost, PoseCost, ProjectedDistCost, JacobianCost, ZeroCost, EEVelCost, StopCost, FiniteDifferenceCost
from ..cost.bound_cost import BoundCost
from ..cost.manipulability_cost import ManipulabilityCost
from ..cost import CollisionCost, VoxelCollisionCost, PrimitiveCollisionCost
from ..model import URDFKinematicModel
from ...util_file import join_path, get_assets_path
from ...differentiable_robot_model.coordinate_transform import CoordinateTransform, matrix_to_quaternion, quaternion_to_matrix
from ...mpc.model.integration_utils import build_int_matrix, build_fd_matrix, tensor_step_acc, tensor_step_vel, tensor_step_pos, tensor_step_jerk


from storm_kit.mpc.rollout.rollout_base import RolloutBase
from storm_kit.mpc.model.simple_pushing_model import SimplePushingModel
from storm_kit.mpc.control.control_utils import cost_to_go
from storm_kit.geom.sim import RigidBody, Material
from storm_kit.geom.shapes import Sphere


class PointRobotPusher(RolloutBase):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update cfg to be kwargs
    """

    def __init__(self, cfg, world_params=None, value_function=None, viz_rollouts=False, device=torch.device('cpu')):
        self.device = device
        self.cfg = cfg
        self.world_params = world_params
        self.world_model = self.world_params["world_model"]

        self.num_instances = cfg['num_instances']
        self.horizon = cfg['horizon']
        self.batch_size = cfg['batch_size']
        self.viz_rollouts = viz_rollouts
        self.value_function = value_function
        self.model = SimplePushingModel(
            batch_size = self.cfg['batch_size'],
            horizon = self.cfg['horizon'],
            num_instances = self.num_instances,
            dt_traj_params = self.cfg['model']['dt_traj_params'], 
            control_space = self.cfg['control_space'],
            robot_keys=['q_pos', 'q_vel', 'q_acc'],
            device = self.device
        )

        #TODO: remove hard coding
        self.object_radius = 0.02
        self.robot_radius = 0.03 #0.01

        table_dims = torch.tensor(self.world_model["coll_objs"]["cube"]["table"]["dims"][0:2], device=self.device)
        self.workspace_dims = torch.zeros(2,2, device=self.device)
        self.workspace_dims[:,0] = -1.0 * table_dims / 2.0
        self.workspace_dims[:,1] = table_dims / 2.0
        self.n_dofs = cfg['n_dofs']
        self.default_object_goal = torch.tensor([0.0, 0.0], device=self.device)
        # self.default_object_init_pos = torch.tensor([-table_dims[0]/2.0 + self.robot_radius + self.object_radius + 0.01, 0.0], device=self.device) #+0.5
        self.default_object_init_pos = torch.tensor([0.0 + self.robot_radius + self.object_radius, 0.0], device=self.device) #+0.5

        self.init_buffers()

        self.object_position_cost = DistCost(
            weight=10.0,
            vec_weight=[1.0, 1.0],
            device=self.device
        )

        self.object_robot_cost = DistCost(
            weight=0.0,
            vec_weight=[1.0, 1.0],
            device=self.device
        )

        self.object_vel_cost = DistCost(
            weight=0.0,
            vec_weight=[1.0, 1.0],
            device=self.device
        )

        self.robot_vel_cost = DistCost(
            weight=0.0,
            vec_weight=[1.0, 1.0],
            device=self.device
        )

        # self.termination_cost_weight = 100.0 # 5000.0
        self.goal_bonus_weight = 100.0
        self.alive_bonus = 1.0 #1.0

        self.vis_initialized = False


    def init_buffers(self):
        # self.robot_pos_buff = torch.zeros(self.num_instances, self.batch_size, self.horizon, 3, device=self.device)
        # self.robot_vel_buff =  torch.zeros(self.num_instances, self.batch_size, self.horizon, 3, device=self.device)
        # self.object_pos_buff =  torch.zeros(self.num_instances, self.batch_size, self.horizon, 3, device=self.device)
        # self.object_vel_buff =  torch.zeros(self.num_instances, self.batch_size, self.horizon, 3, device=self.device)
        self.object_goal_buff = torch.zeros(self.num_instances, 2, device=self.device)
        self.object_init_pos_buff = torch.zeros(self.num_instances, 2, device=self.device)
        self.value_preds_buff = torch.zeros(self.num_instances, self.batch_size, self.horizon, device=self.device)

    def compute_cost(self, 
            state_dict: Dict[str, torch.Tensor], 
            action_batch: Optional[Dict[str, torch.Tensor]]=None,
            termination: Optional[Dict[str, torch.Tensor]]=None):
        
        object_pos = state_dict['object_pos'][..., 0:2].view(self.num_instances, self.batch_size, self.horizon, 2)
        object_vel = state_dict['object_vel'][..., 0:2].view(self.num_instances, self.batch_size, self.horizon, 2)
        goal_object_pos = self.object_goal_buff.unsqueeze(1).unsqueeze(1).repeat(1, self.batch_size, self.horizon, 1)
        start_object_pos = self.object_init_pos_buff.unsqueeze(1).unsqueeze(1).repeat(1, self.batch_size, self.horizon, 1)
        # goal_object_pos = goal_object_pos.view(self.num_instances * self.batch_size, self.horizon, 2)
        robot_pos = state_dict['q_pos'][..., 0:2].view(self.num_instances, self.batch_size, self.horizon, 2)
        robot_vel = state_dict['q_vel'][..., 0:2].view(self.num_instances, self.batch_size, self.horizon, 2)


        # object_pos_cost_rel, dist = self.object_position_cost(
        #     object_pos - goal_object_pos, 
        #     norm_vec = start_object_pos - goal_object_pos)

        object_pos_cost, dist = self.object_position_cost(
            object_pos - goal_object_pos)


        cost_terms = {
            # 'object_pos_rel': object_pos_cost_rel,
            'object_pos': object_pos_cost,
            # 'object_robot_rel_pos': obj_robot_cost,
        }

        # in_goal_tolerance = dist <= 0.01
        # goal_bonus = self.goal_bonus_weight * in_goal_tolerance
        # cost_terms['goal_bonus'] = goal_bonus

        # obj_robot_cost, _ = self.object_robot_cost(robot_pos - object_pos)
        # cost_terms['object_robot_rel_pos'] = obj_robot_cost
        
        obj_vel_cost, _ = self.object_vel_cost(object_vel)
        # obj_vel_cost[dist >= 0.01] = 0.0
        cost_terms['object_vel_cost'] = obj_vel_cost

        robot_vel_cost, _ = self.robot_vel_cost(robot_vel)
        # robot_vel_cost[dist >= 0.01] = 0.0
        cost_terms['robot_vel_cost'] = robot_vel_cost


        # cost = object_pos_cost_rel + object_pos_cost_abs + obj_vel_cost + robot_vel_cost #+ obj_robot_cost
        cost = object_pos_cost + obj_vel_cost + robot_vel_cost #- goal_bonus #+ obj_robot_cost


        if termination is not None:
            # termination_cost = self.termination_cost_weight * termination 
            alive_bonus = 1.0 * self.alive_bonus * (1.0 - termination.float())
            # cost += termination_cost
            cost -= alive_bonus
            # cost_terms['termination'] = termination_cost
            cost_terms['alive_bonus'] = alive_bonus


        return cost, cost_terms, state_dict

    def compute_observations(self, 
                             state_dict: Dict[str,torch.Tensor]):

        object_goal_buff = self.object_goal_buff.clone()
        object_start_buff = self.object_init_pos_buff.clone()

        if state_dict['q_pos'].ndim > 2:
            _, batch_size, horizon, _ = state_dict['q_pos'].shape
            object_goal_buff = self.object_goal_buff.unsqueeze(1).unsqueeze(1).repeat(1, batch_size, horizon, 1)
            object_start_buff = self.object_init_pos_buff.unsqueeze(1).unsquueze(1).repeat(1, batch_size, horizon, 1)

        object_pos = state_dict['object_pos']
        object_vel = state_dict['object_vel']
        robot_pos = state_dict['q_pos']
        robot_vel = state_dict['q_vel']

        obs = torch.cat((
            robot_pos, robot_vel, object_pos, object_vel, 
            object_goal_buff, object_goal_buff - object_pos,
            # object_start_buff, object_start_buff - object_pos,
            robot_pos - object_pos,
            ), dim=-1)

        return obs, state_dict

    def compute_termination(self, state_dict: Dict[str,torch.Tensor], act_batch: Dict[str,torch.Tensor]):
        object_pos = state_dict['object_pos']
        robot_pos = state_dict['q_pos']
        x_lims = self.workspace_dims[0]
        y_lims = self.workspace_dims[1]
        obj_x = object_pos[..., 0] 
        obj_y = object_pos[..., 1]
        robot_x = robot_pos[..., 0] 
        robot_y = robot_pos[..., 1]
        #termination based on object
        termination = (obj_x - self.object_radius <= x_lims[0]) + (obj_x + self.object_radius >= x_lims[1])
        termination += (obj_y - self.object_radius <= y_lims[0]) + (obj_y + self.object_radius >= y_lims[1])
        #termination based on robot
        termination += (robot_x - self.robot_radius <= x_lims[0]) + (robot_x + self.robot_radius >= x_lims[1])
        termination += (robot_y - self.robot_radius <= y_lims[0]) + (robot_y + self.robot_radius >= y_lims[1])
        #resolve all termination conditions
        termination = (termination > 0)
        return termination, state_dict
    
    def compute_value_predictions(self, state_dict: torch.Tensor, act_seq):
        value_preds = None
        if self.value_function is not None:
            obs, state_dict = self.compute_observations(state_dict)
            input_dict = {
                'obs': obs,
                'states': state_dict
            }
            value_preds = -1.0 * self.value_function.forward(input_dict, act_seq) #value function is trained to maximize rewards

        return value_preds, state_dict

    def compute_metrics(self, episode_data: Dict[str, torch.Tensor]):
        states = episode_data['state_dict']
        object_pos = states['object_pos']
        object_goal = episode_data['goal_dict']['object_goal']

        errors = torch.norm(object_pos - object_goal, p=2, dim=-1)
        init_error = errors[0]
        last_n_errors = errors[-5:]
        normalized_errors = last_n_errors / init_error
        # normalized_errors = errors / init_error

        normalized_error_sum = torch.sum(normalized_errors).item()
        normalized_error_mean = torch.mean(normalized_errors).item()
        normalized_error_std = torch.std(normalized_errors).item()

        return {
            'normalized_errors_sum': normalized_error_sum,
            'mean_normalized_errors': normalized_error_mean, 
            'std_normalized_errors': normalized_error_std}


    def rollout_fn(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Parameters
        ----------
        action_seq: torch.Tensor [num_particles, horizon, d_act]
        """

        with record_function("robot_model"):
            state_dict = self.model.rollout_open_loop(start_state, act_seq)

        with record_function("compute_termination"):
            term_seq, _ = self.compute_termination(state_dict, act_seq)
        
        with record_function("cost_fns"):
            cost_seq, _, _ = self.compute_cost(state_dict, act_seq, termination=term_seq)
        
        with record_function("value_fn_inference"):
            value_preds, _ = self.compute_value_predictions(state_dict, act_seq)

        sim_trajs = dict(
            states = state_dict,
            actions=act_seq, #.clone(),
            costs=cost_seq, #clone(),
            terminations=term_seq,
            value_preds=value_preds,
            rollout_time=0.0
        )

        if self.viz_rollouts:
            self.visualize_rollouts(sim_trajs)
        
        return sim_trajs

    # def compute_distance_squared(self, robot_pos, object_pos):
    #     robot_x = robot_pos[:,:,0]
    #     robot_y = robot_pos[:,:,1]
    #     object_x = object_pos[:,:,0]
    #     object_y = object_pos[:,:,1]

    #     return (robot_x - object_x) ** 2 + (robot_y - object_y) ** 2

    # def compute_collision(self, robot_pos, object_pos):
    #     r = self.robot_radius + self.object_radius
    #     r *= r
    #     dist_squared = self.compute_distance_squared(robot_pos, object_pos) 
    #     in_coll =  r >= dist_squared
    #     return in_coll.float(), dist_squared 

    def update_params(self, param_dict):
        """
        Updates the goal targets for the cost functions.

        """
        if 'goal_dict' in param_dict:
            self.object_goal_buff = param_dict['goal_dict']['object_goal']
        if 'start_dict' in param_dict:
            self.object_init_pos_buff = param_dict['start_dict']['object_start_pos']
            
    def reset(self):
        env_ids = torch.arange(self.num_instances, device=self.device)
        return self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        # self.robot_pos_buff[env_ids] = torch.zeros(len(env_ids), self.batch_size, self.horizon, 3, device=self.device)
        # self.robot_vel_buff[env_ids] = torch.zeros(len(env_ids), self.batch_size, self.horizon, 3, device=self.device)
        # self.object_pos_buff[env_ids] = torch.zeros(len(env_ids), self.batch_size, self.horizon, 3, device=self.device)
        # self.object_vel_buff[env_ids] = torch.zeros(len(env_ids), self.batch_size, self.horizon, 3, device=self.device)

        #randomize goals
        self.object_goal_buff[env_ids] = self.default_object_goal
        # self.object_goal_buff[env_ids, 0] += 0.2
        # self.object_goal_buff[env_ids, 1] += 0.2 * torch.rand(1, device=self.device) - 0.1

        self.object_goal_buff[env_ids, 0] += 0.4 * torch.rand(1, device=self.device) - 0.2
        self.object_goal_buff[env_ids, 1] += 0.4 * torch.rand(1, device=self.device) - 0.2

        #randomize object positions
        self.object_init_pos_buff[env_ids] = self.default_object_init_pos

        #randomize ball location
        reset_data = {}
        reset_data['goal_dict'] = {
            'object_goal': self.object_goal_buff}
        reset_data['start_dict'] = {
            'object_start_pos': self.object_init_pos_buff
        }
        return reset_data


    def init_viewer(self):
        # if not self.vis_initialized:
        #     print("herer")
        #     import matplotlib.pyplot as plt
        #     self.fig, self. ax = plt.subplots()
        #     plt.show(block=False)
        #     self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        #     self.fig.canvas.blit(self.fig.bbox)
        #     self.vis_initialized = True
        
        if not self.vis_initialized:
            print('Initializing rollout viewer')
            import meshcat
            import meshcat.geometry as meshcat_g

            self.vis = meshcat.Visualizer() #if self.vis is None else self.vis
            # self.vis.open()
            


            self.vis_initialized = True

            for h in range(self.horizon):
                robot_material = meshcat_g.MeshBasicMaterial(
                    color=0xff0000, transparent=True, opacity=1.0 / (1.0 + 0.1*h))
                object_material = meshcat_g.MeshBasicMaterial(
                    color=0x0000FF, transparent=True, opacity=1.0 / (1.0 + 0.1*h))
                goal_material = meshcat_g.MeshBasicMaterial(
                    color=0x00FF00, transparent=True, opacity=1.0 / (1.0 + 0.1*h))
                
                self.vis["world"]["robot"][str(h)].set_object(meshcat_g.Sphere(self.robot_radius), robot_material)
                self.vis["world"]["object"][str(h)].set_object(meshcat_g.Sphere(self.object_radius), object_material)
                self.vis["world"]["goal"][str(h)].set_object(meshcat_g.Sphere(self.object_radius), goal_material)
    
    def visualize_rollouts(self, rollout_data):
        self.init_viewer()
            # self.fig.canvas.restore_region(self.bg)
        import meshcat.transformations as meshcat_tf

        robot_pos = rollout_data['states']['q_pos']
        object_pos = rollout_data['states']['object_pos']
        costs = rollout_data['costs']
        _, _, horizon = costs.shape
        gamma_seq = torch.cumprod(torch.tensor([1.0] + [0.99] * (horizon - 1)),dim=0).reshape(1, horizon)
        gamma_seq = gamma_seq.to(self.device)
        total_costs = cost_to_go(costs, gamma_seq)[:, :, 0]
        top_values, top_idx = torch.topk(total_costs, 10, dim=-1)
        top_idx = top_idx.squeeze(0)

        top_robot_pos = torch.index_select(robot_pos, 1, top_idx).squeeze(0).cpu() #.squeeze(0)
        top_object_pos = torch.index_select(object_pos, 1, top_idx).squeeze(0).cpu()
        top_robot_pos = torch.cat((top_robot_pos, torch.zeros(10, horizon, 1)), dim=-1)
        top_object_pos = torch.cat((top_object_pos, torch.zeros(10, horizon, 1)), dim=-1)

        object_goal = self.object_goal_buff.clone().cpu()
        object_goal = torch.cat((object_goal, torch.zeros(self.num_instances, 1)), dim=-1).numpy()

        for i in range(horizon):
            self.vis["world"]["robot"][str(i)].set_transform(meshcat_tf.translation_matrix(top_robot_pos[0,i]))
            self.vis["world"]["object"][str(i)].set_transform(meshcat_tf.translation_matrix(top_object_pos[0,i]))
            self.vis["world"]["goal"][str(i)].set_transform(meshcat_tf.translation_matrix(object_goal[0]))
        time.sleep(0.01)


    @property
    def obs_dim(self)->int:
        return 14
    
    @property
    def action_dim(self)->int:
        return self.n_dofs

    @property
    def action_lims(self)->Tuple[torch.Tensor, torch.Tensor]:
        act_highs = torch.tensor([self.cfg['model']['max_acc']] * self.action_dim,  device=self.device)
        act_lows = torch.tensor([-1.0 * self.cfg['model']['max_acc']] * self.action_dim, device=self.device)
        return act_lows, act_highs


    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)