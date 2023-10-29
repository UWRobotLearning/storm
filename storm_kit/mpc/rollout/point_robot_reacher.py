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
import matplotlib.pyplot as plt

from ..cost import NormCost
from ..cost.bound_cost import BoundCost
from ...util_file import join_path, get_assets_path
from ..model.integration_utils import build_int_matrix, build_fd_matrix, tensor_step_acc, tensor_step_vel, tensor_step_pos, tensor_step_jerk


from storm_kit.mpc.rollout.rollout_base import RolloutBase
from storm_kit.mpc.model.double_integrator_model import DoubleIntegratorModel
from storm_kit.mpc.control.control_utils import cost_to_go


class PointRobotReacher(RolloutBase):
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
        self.n_dofs = cfg['n_dofs']
        self.viz_rollouts = viz_rollouts
        self.value_function = value_function
        self.model = DoubleIntegratorModel(
            n_dofs = self.n_dofs,
            batch_size = self.cfg['batch_size'],
            horizon = self.cfg['horizon'],
            num_instances = self.num_instances,
            dt_traj_params = self.cfg['model']['dt_traj_params'], 
            control_space = self.cfg['control_space'],
            robot_keys=['q_pos', 'q_vel', 'q_acc'],
            device = self.device
        )

        self.robot_radius = 0.03

        table_dims = torch.tensor(self.world_model["coll_objs"]["cube"]["table"]["dims"][0:2], device=self.device)
        self.workspace_dims = torch.zeros(2,2, device=self.device)
        self.workspace_dims[:,0] = -1.0 * table_dims / 2.0
        self.workspace_dims[:,1] = table_dims / 2.0
        self.task_specs = cfg['task_specs']
        self.default_ee_goal = torch.tensor(self.task_specs['default_ee_target'], device=self.device)
        self.default_robot_init_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        self.init_buffers()

        self.goal_cost = NormCost(
            weight=50.0,
            norm_type='squared_l2',
            device=self.device
        )

        self.robot_vel_cost = DistCost(
            weight=0.0,
            norm_type='squared_l2',
            device=self.device
        )

        # self.goal_bonus_weight = 100.0
        self.termination_cost_weight = 1.0
        # self.alive_bonus = 1.0 #1.0
        self.vis_initialized = False


    def init_buffers(self):
        self.ee_goal_buff = torch.zeros(self.num_instances, 7, device=self.device)
        self.robot_init_pos_buff = torch.zeros(self.num_instances, 2, device=self.device)
        self.value_preds_buff = torch.zeros(self.num_instances, self.batch_size, self.horizon, device=self.device)
        self.termination_buff = torch.zeros(self.num_instances, self.batch_size, self.horizon, device=self.device)

    def compute_cost(self, 
            state_dict: Dict[str, torch.Tensor], 
            action_batch: Optional[Dict[str, torch.Tensor]]=None,
            termination: Optional[Dict[str, torch.Tensor]]=None):
        
        goal_robot_pos = self.ee_goal_buff[:, 0:3].unsqueeze(1).unsqueeze(1).repeat(1, self.batch_size, self.horizon, 1)
        robot_pos = state_dict['q_pos']#.view(self.num_instances, self.batch_size, self.horizon, -1)
        robot_vel = state_dict['q_vel']#.view(self.num_instances, self.batch_size, self.horizon, -1)

        goal_cost, dist = self.goal_cost(
            robot_pos - goal_robot_pos)


        cost_terms = {
            'goal': goal_cost,
        }

        robot_vel_cost, _ = self.robot_vel_cost(robot_vel)
        # robot_vel_cost[dist >= 0.01] = 0.0
        cost_terms['robot_vel_cost'] = robot_vel_cost

        cost =  goal_cost + robot_vel_cost


        if termination is not None:
            termination_cost = self.termination_cost_weight * termination 
            cost += termination_cost
            cost_terms['termination'] = termination_cost
            # alive_bonus = 1.0 * self.alive_bonus * (1.0 - termination.float())
            # cost -= alive_bonus
            # cost_terms['alive_bonus'] = alive_bonus


        return cost, cost_terms, state_dict

    def compute_observations(self, 
                             state_dict: Dict[str,torch.Tensor]):

        ee_goal_buff = self.ee_goal_buff.clone()

        if state_dict['q_pos'].ndim > 2:
            _, batch_size, horizon, _ = state_dict['q_pos'].shape
            ee_goal_buff = self.ee_goal_buff.unsqueeze(1).unsqueeze(1).repeat(1, batch_size, horizon, 1)

        robot_pos = state_dict['q_pos']
        robot_vel = state_dict['q_vel']

        obs = torch.cat((
            robot_pos, robot_vel,
            ee_goal_buff[:,0:3], ee_goal_buff[:, 0:3] - robot_pos,
            ), dim=-1)

        return obs, state_dict

    def compute_termination(self, state_dict: Dict[str,torch.Tensor], act_batch: Dict[str,torch.Tensor]):
        #termination is based on robot position in sim coordinates
        # robot_pos = state_dict['q_pos']
        # x_lims = self.workspace_dims[0]
        # y_lims = self.workspace_dims[1]
        # robot_x = robot_pos[..., 0] 
        # robot_y = robot_pos[..., 1]
        # #termination based on robot
        # termination = (robot_x - self.robot_radius <= x_lims[0]) + (robot_x + self.robot_radius >= x_lims[1])
        # termination += (robot_y - self.robot_radius <= y_lims[0]) + (robot_y + self.robot_radius >= y_lims[1])
        # #resolve all termination conditions
        # termination = (termination > 0)
        # return termination, state_dict
        return self.termination_buff, state_dict
    
    def compute_value_predictions(self, state_dict: torch.Tensor, act_seq):
        value_preds = None
        if self.value_function is not None:
            obs, state_dict = self.compute_observations(state_dict)
            input_dict = {
                'obs': obs,
                'states': state_dict
            }
            value_preds = self.value_function.forward(input_dict, act_seq)

        return value_preds, state_dict

    def compute_metrics(self, episode_data: Dict[str, torch.Tensor]):
        states = episode_data['state_dict']
        robot_pos = states['q_pos']
        robot_goal = episode_data['goal_dict']['ee_goal'][:, 0:3]

        errors = torch.norm(robot_pos - robot_goal, p=2, dim=-1)
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


    def update_params(self, param_dict):
        """
        Updates the goal targets for the cost functions.

        """
        if 'goal_dict' in param_dict:
            self.ee_goal_buff = param_dict['goal_dict']['ee_goal']
        if 'start_dict' in param_dict:
            self.robot_init_pos_buff = param_dict['start_dict']['robot_start_pos']
            
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
            # self.ee_goal_buff[env_ids, 2] = self.ee_goal_buff[env_ids, 2] +  2.0*goal_position_noise * (torch.rand_like(self.ee_goal_buff[env_ids, 2], device=self.device) - 0.5)

        if goal_rotation_noise > 0.:
            #randomize goal orientation
            raise NotImplementedError('orientation randomization not implemented')

        # #randomize robotositions
        # self.robot_init_pos_buff[env_ids] = self.default_robot_init_pos

        #randomize ball location
        reset_data = {}
        reset_data['goal_dict'] = {
            'ee_goal': self.ee_goal_buff}
        # reset_data['start_dict'] = {
        #     'robot_start_pos': self.robot_init_pos_buff
        # }
        return reset_data


    def init_viewer(self):

        if not self.vis_initialized:
            print('Initializing rollout viewer')
            import meshcat
            import meshcat.geometry as meshcat_g

            self.vis = meshcat.Visualizer() #if self.vis is None else self.vis
            self.vis_initialized = True

            for h in range(self.horizon):
                robot_material = meshcat_g.MeshBasicMaterial(
                    color=0xff0000, transparent=True, opacity=1.0 / (1.0 + 0.1*h))
                goal_material = meshcat_g.MeshBasicMaterial(
                    color=0x00FF00, transparent=True, opacity=1.0 / (1.0 + 0.1*h))
                
                self.vis["world"]["robot"][str(h)].set_object(meshcat_g.Sphere(self.robot_radius), robot_material)
                self.vis["world"]["goal"][str(h)].set_object(meshcat_g.Sphere(self.robot_radius), goal_material)

            self.fig, self.ax = plt.subplots(self.action_dim)
        
    
    def visualize_rollouts(self, rollout_data):
        self.init_viewer()
            # self.fig.canvas.restore_region(self.bg)
        import meshcat.transformations as meshcat_tf

        robot_pos = rollout_data['states']['q_pos']
        costs = rollout_data['costs']
        _, _, horizon = costs.shape
        gamma_seq = torch.cumprod(torch.tensor([1.0] + [0.99] * (horizon - 1)),dim=0).reshape(1, horizon)
        gamma_seq = gamma_seq.to(self.device)
        total_costs = cost_to_go(costs, gamma_seq)[:, :, 0]
        top_values, top_idx = torch.topk(total_costs, 10, dim=-1)
        top_idx = top_idx.squeeze(0)

        top_robot_pos = torch.index_select(robot_pos, 1, top_idx).squeeze(0).cpu() #.squeeze(0)
        top_robot_pos = torch.cat((top_robot_pos, torch.zeros(10, horizon, 1)), dim=-1)

        robot_goal = self.ee_goal_buff.clone().cpu()
        robot_goal = torch.cat((robot_goal, torch.zeros(self.num_instances, 1)), dim=-1).numpy()

        for i in range(horizon):
            self.vis["world"]["robot"][str(i)].set_transform(meshcat_tf.translation_matrix(top_robot_pos[0,i]))
            self.vis["world"]["goal"][str(i)].set_transform(meshcat_tf.translation_matrix(robot_goal[0]))
        
        #Pliot the actions as well
        actions = rollout_data['actions'].cpu().numpy()
        _, b, h, nd = actions.shape 
            # fig, ax = plt.subplots(nd)

        for d_i in range(nd):
            self.ax[d_i].clear()
            for b_i in range(b):
                data = actions[0, b_i, :, d_i]
                self.ax[d_i].plot(data)
        plt.pause(0.01)
        plt.draw()

    @property
    def obs_dim(self)->int:
        return 12
    
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