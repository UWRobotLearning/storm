from typing import Dict, Optional, Tuple
import torch
from torch.profiler import record_function
from functools import reduce
from operator import mul

from storm_kit.differentiable_robot_model.spatial_vector_algebra import matrix_to_quaternion, quaternion_to_matrix, euler_angles_to_matrix, matrix_to_euler_angles
from storm_kit.mpc.cost import NormCost, PoseCost
from storm_kit.tasks.arm_reacher import ArmReacher

class TrayReacher(ArmReacher):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update cfg to be kwargs
    """

    def __init__(self, cfg, world_params=None, viz_rollouts=False, device=torch.device('cpu')):
        super(TrayReacher, self).__init__(
            cfg=cfg,
            world_params=world_params,
            viz_rollouts=viz_rollouts,
            device=device)

    def forward(self, state_dict: Dict[str, torch.Tensor], act_batch:Optional[torch.Tensor]=None):
        return super().forward(state_dict, act_batch)

    def compute_observations(self, state_dict: Dict[str, torch.Tensor], compute_full_state:bool = False, debug=False):

        if compute_full_state:
            state_dict = self._compute_full_state(state_dict, debug)
        
        obs =  super().compute_observations(state_dict)

        return obs

    def compute_cost(
            self, 
            state_dict: Dict[str, torch.Tensor], 
            action_batch: Optional[torch.Tensor]=None,
            horizon_cost: bool=True, return_dist:bool=False):

        cost, cost_terms = super().compute_cost(
            state_dict = state_dict,
            action_batch = action_batch,
            horizon_cost = horizon_cost)

        return cost, cost_terms #, state_dict


    def compute_termination(self, state_dict: Dict[str,torch.Tensor], act_batch: torch.Tensor):
        return super().compute_termination(state_dict, act_batch)


    def compute_metrics(self, episode_data: Dict[str, torch.Tensor]):
        q_pos = episode_data['state_dict']['q_pos'].to(self.device)
        ee_goal = episode_data['goal_dict']['ee_goal'].to(self.device)
        ee_goal_pos = ee_goal[:, 0:3]
        ee_goal_quat = ee_goal[:, 3:7]
        ee_goal_rot = quaternion_to_matrix(ee_goal_quat)
        ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.robot_model.compute_fk_and_jacobian(
            q_pos, self.cfg['ee_link_name'])

        goal_cost, rot_err, dist_err = self.goal_cost.forward(
            ee_pos_batch, ee_rot_batch,
            ee_goal_pos, ee_goal_rot) #jac_batch=J_full
        
        #Last Step Error{'num_instances': 1, 'control_space': '${task.task.control_space}', 'max_acc': '${task.task.max_acc}', 'n_dofs': '${task.n_dofs}', 'model': {'urdf_path': '${task.robot_urdf}', 'learnable_rigid_body_config': {'learnable_links': []}, 'name': 'franka_panda', 'dt_traj_params': {'base_dt': 0.02, 'base_ratio': 1.0, 'max_dt': 0.2}, 'ee_link_name': '${task.ee_link_name}', 'init_state': [0.8, 0.3, 0.0, -1.57, 0.0, 1.86, 0.0], 'link_names': '${task.task.robot_link_names}', 'collision_spheres': '../robot/franka.yml'}, 'robot_collision_params': '${task.task.robot_collision_params}', 'world_collision_params': '${task.task.world_collision_params}', 'cost': {'goal_pose': {'vec_weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'weight': [0.0, 0.0], 'cost_type': 'se3_twist', 'norm_type': 'l1', 'hinge_val': -1, 'convergence_val': [0.0, 0.0]}, 'ee_vel_twist': {'weight': 0.003, 'norm_type': 'l1'}, 'ee_acc_twist': {'weight': 0.0, 'norm_type': 'l1'}, 'zero_q_vel': {'weight': 0.01, 'norm_type': 'l1'}, 'zero_q_acc': {'weight': 0.0, 'norm_type': 'l1'}, 'zero_q_jerk': {'weight': 0.0, 'norm_type': 'l1'}, 'manipulability': {'weight': 0.5, 'thresh': 0.1}, 'joint_l2': {'weight': 10.0, 'norm_type': 'l1'}, 'stop_cost': {'weight': 1.5, 'max_nlimit': 5.0}, 'stop_cost_acc': {'weight': 0.0, 'max_limit': 0.1}, 'smooth_cost': {'weight': 0.0, 'order': 1}, 'primitive_collision': {'weight': 100.0, 'distance_threshold': 0.03}, 'state_bound': {'weight': 100.0, 'bound_thresh': 0.03}, 'retract_state': [0.0, 0.0, 0.0, -1.5, 0.0, 2.0, 0.0], 'retract_weight': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, 'horizon': 20, 'batch_size': 500, 'dt_traj_params': {'base_dt': 0.02, 'base_ratio': 1.0, 'max_dt': 0.2}}
        final_goal_dist_err = dist_err[-1].item()
        final_goal_rot_err = rot_err[-1].item()
        final_goal_dist_err_rel = final_goal_dist_err / dist_err[0].item()
        final_goal_rot_err_rel = final_goal_rot_err / dist_err[0].item()

        #Last 10 step avg error
        last_n_dist_err =  torch.mean(dist_err[-10:]).item()
        last_n_dist_err_rel = last_n_dist_err / dist_err[0].item()
        
        #Max and average joint velocity
        q_vel = episode_data['state_dict']['q_vel'].to(self.device)
        q_vel_norm = torch.norm(q_vel, p=2, dim=-1)
        max_q_vel = torch.max(q_vel_norm).item()
        avg_q_vel = torch.mean(q_vel_norm).item()

        #EE velocity
        ee_vel = torch.matmul(lin_jac_batch, q_vel.unsqueeze(-1)).squeeze(-1)
        ee_ang_vel = torch.matmul(ang_jac_batch, q_vel.unsqueeze(-1)).squeeze(-1)
        ee_vel_norm = torch.norm(ee_vel, p=2, dim=-1)
        ee_ang_vel_norm = torch.norm(ee_ang_vel, p=2, dim=-1)
        max_ee_vel = torch.max(ee_vel_norm).item()
        max_ee_ang_vel = torch.max(ee_ang_vel_norm).item()
        avg_ee_vel = torch.mean(ee_vel_norm).item()
        avg_ee_ang_vel = torch.mean(ee_ang_vel_norm).item()

        return {
            'final_goal_dist_err': final_goal_dist_err,
            'final_goal_rot_err': final_goal_rot_err,
            'final_goal_dist_err_rel': final_goal_dist_err_rel,
            'final_goal_rot_err_rel': final_goal_rot_err_rel,
            'last_10_dist_err': last_n_dist_err,
            'last_10_dist_err_rel': last_n_dist_err_rel,
            'max_q_vel': max_q_vel,
            'avg_q_vel': avg_q_vel,
            'max_ee_vel': max_ee_vel,
            'max_ee_ang_vel': max_ee_ang_vel,
            'avg_ee_vel': avg_ee_vel,
            'avg_ee_ang_vel': avg_ee_ang_vel}

    def reset(self):
        env_ids = torch.arange(self.num_instances, device=self.device)
        return self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        return super().reset_idx(env_ids)

    def update_params(self, param_dict):        
        return  super().update_params(param_dict=param_dict)
        
    @property
    def obs_dim(self)->int:
        return super().obs_dim 


