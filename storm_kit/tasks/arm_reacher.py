from typing import Dict, Optional, Tuple
import torch
from torch.profiler import record_function
from functools import reduce
from operator import mul

from storm_kit.differentiable_robot_model.spatial_vector_algebra import matrix_to_quaternion, quaternion_to_matrix, euler_angles_to_matrix, matrix_to_euler_angles
from storm_kit.mpc.cost import NormCost, PoseCost
from storm_kit.tasks.arm_task import ArmTask

class ArmReacher(ArmTask):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update cfg to be kwargs
    """

    def __init__(self, cfg, world_params=None, viz_rollouts=False, device=torch.device('cpu')):
        super(ArmReacher, self).__init__(cfg=cfg,
                                         world_params=world_params,
                                         viz_rollouts=viz_rollouts,
                                         device=device)
        self.goal_state = None
        self.goal_ee_pos = None
        self.goal_ee_rot = None
        
        self.dist_cost = NormCost(**self.cfg['cost']['joint_l2'], device=self.device)
        self.goal_cost = PoseCost(**self.cfg['cost']['goal_pose'], device=self.device)

        self.task_specs = cfg.get('task_specs', None)
        if self.task_specs is not None:
            self.default_ee_goal = torch.tensor(self.task_specs['default_ee_target'], device=self.device)
        self.init_buffers()

    def init_buffers(self):
        self.ee_goal_buff = torch.zeros(self.num_instances, 7, device=self.device)
        self.prev_state_buff = torch.zeros(self.num_instances, 10, 22, device=self.device)
    
    def forward(self, state_dict: Dict[str, torch.Tensor], act_batch:Optional[torch.Tensor]=None):
        return super().forward(state_dict, act_batch)

    def compute_observations(self, state_dict: Dict[str, torch.Tensor], compute_full_state:bool = False, debug=False):

        if compute_full_state:
            state_dict = self._compute_full_state(state_dict, debug)
        
        obs =  super().compute_observations(state_dict)

        # orig_size = state_dict['q_pos_seq'].size()[0:-1]
        # new_size = reduce(mul, list(orig_size))  
        # obs = obs.view(new_size, -1)

        ee_goal_pos = self.goal_ee_pos
        ee_goal_quat = self.goal_ee_quat
        if obs.ndim > 2:
            target_shape = list(obs.shape[0:-1])
            ee_goal_pos = ee_goal_pos.view(ee_goal_pos.shape[0], *(1,)*(len(target_shape)-ee_goal_pos.ndim), ee_goal_pos.shape[-1]).expand(target_shape)
            ee_goal_quat = ee_goal_quat.view(ee_goal_quat.shape[0], *(1,)*(len(target_shape)-ee_goal_quat.ndim), ee_goal_quat.shape[-1]).expand(target_shape)

        obs = torch.cat((obs, ee_goal_pos, ee_goal_quat), dim=-1)

        return obs

    def compute_cost(
            self, 
            state_dict: Dict[str, torch.Tensor], 
            action_batch: Optional[torch.Tensor]=None,
            horizon_cost: bool=True, return_dist:bool=False):

        cost, cost_terms = super(ArmReacher, self).compute_cost(
            state_dict = state_dict,
            action_batch = action_batch,
            horizon_cost = horizon_cost)

        # num_instances, curr_batch_size, num_traj_points, _ = state_dict['state_seq'].shape
        # cost = cost.view(num_instances, curr_batch_size, num_traj_points)
        q_pos_batch = state_dict['q_pos_seq']
        orig_size = q_pos_batch.size()[0:-1]
        new_size = reduce(mul, list(orig_size))  

        ee_pos_batch, ee_rot_batch = state_dict['ee_pos_seq'].view(new_size, 3), state_dict['ee_rot_seq'].view(new_size, 3, 3)
        # lin_jac_batch, ang_jac_batch = state_dict['lin_jac_seq'], state_dict['ang_jac_seq']
        # lin_jac_batch = lin_jac_batch.view(self.num_instances*self.batch_size, self.horizon, 3, self.n_dofs)
        # ang_jac_batch = ang_jac_batch.view(self.num_instances*self.batch_size, self.horizon, 3, self.n_dofs)
        # ee_jacobian_seq = state_dict['ee_jacobian_seq'].view(self.num_instances*self.batch_size, self.horizon, 6, self.n_dofs)

        state_batch = state_dict['state_seq']#.view(num_instances*curr_batch_size, num_traj_points, -1)

        goal_ee_pos = self.goal_ee_pos
        goal_ee_rot = self.goal_ee_rot
        goal_state = self.goal_state


        # with record_function("pose_cost"):
        #     goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward(ee_pos_batch, ee_rot_batch,
        #                                                                 goal_ee_pos, goal_ee_rot)

        with record_function("pose_cost"):
            goal_ee_pos = goal_ee_pos.repeat(ee_pos_batch.shape[0] // self.num_instances, 1)
            goal_ee_rot = goal_ee_rot.repeat(ee_rot_batch.shape[0] // self.num_instances, 1,1)

            goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward(ee_pos_batch, ee_rot_batch,
                                                                        goal_ee_pos, goal_ee_rot) #jac_batch=J_full
            # goal_cost[:,:,0:-1] = 0.
        cost += goal_cost.view(orig_size)
        
        # joint l2 cost
        if self.cfg['cost']['joint_l2']['weight'] > 0.0 and goal_state is not None:
            disp_vec = state_batch[..., 0:self.n_dofs] - goal_state[...,0:self.n_dofs]
            dist_cost, _ = self.dist_cost.forward(disp_vec)
            cost += dist_cost.view(orig_size) #self.dist_cost.forward(disp_vec)

        if return_dist:
            return cost, state_dict, rot_err_norm, goal_dist
        
        return cost, cost_terms #, state_dict


    def compute_termination(self, state_dict: Dict[str,torch.Tensor], act_batch: torch.Tensor):
        return super().compute_termination(state_dict, act_batch)


    def compute_metrics(self, episode_data: Dict[str, torch.Tensor]):
        q_pos = episode_data['state_dict']['q_pos']
        ee_goal = episode_data['goal_dict']['ee_goal']
        ee_goal_pos = ee_goal[:, 0:3]
        ee_goal_quat = ee_goal[:, 3:7]
        ee_goal_rot = quaternion_to_matrix(ee_goal_quat)
        ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.robot_model.compute_fk_and_jacobian(
            q_pos, self.cfg['ee_link_name'])

        goal_cost, rot_err, dist_err = self.goal_cost.forward(
            ee_pos_batch, ee_rot_batch,
            ee_goal_pos, ee_goal_rot) #jac_batch=J_full
        
        #Last Step Error
        final_goal_dist_err = dist_err[-1].item()
        final_goal_rot_err = rot_err[-1].item()
        final_goal_dist_err_rel = final_goal_dist_err / dist_err[0].item()
        final_goal_rot_err_rel = final_goal_rot_err / dist_err[0].item()

        #Last 10 step avg error
        last_n_dist_err =  torch.mean(dist_err[-10:]).item()
        last_n_dist_err_rel = last_n_dist_err / dist_err[0].item()
        
        #Max and average joint velocity
        q_vel = episode_data['state_dict']['q_vel']
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
        reset_data = {}
        if self.task_specs is not None:
            goal_position_noise = self.task_specs['target_position_noise']
            goal_rotation_noise = self.task_specs['target_rotation_noise']

            self.ee_goal_buff[env_ids] = self.default_ee_goal

            if goal_position_noise > 0.:
                #randomize goal position around the default
                self.ee_goal_buff[env_ids, 0] = self.ee_goal_buff[env_ids, 0] +  2.0*goal_position_noise * (torch.rand_like(self.ee_goal_buff[env_ids, 0], device=self.device) - 0.5)
                self.ee_goal_buff[env_ids, 1] = self.ee_goal_buff[env_ids, 1] +  2.0*goal_position_noise * (torch.rand_like(self.ee_goal_buff[env_ids, 1], device=self.device) - 0.5)
                self.ee_goal_buff[env_ids, 2] = self.ee_goal_buff[env_ids, 2] +  2.0*goal_position_noise * (torch.rand_like(self.ee_goal_buff[env_ids, 2], device=self.device) - 0.5)
            
            if goal_rotation_noise > 0.:
                #randomize goal orientation around defualt
                default_quat = self.default_ee_goal[3:7]
                default_euler = matrix_to_euler_angles(quaternion_to_matrix(default_quat).unsqueeze(0), convention='XYZ')
                roll = default_euler[:,0] + 2.0 * goal_rotation_noise * (torch.rand(env_ids.shape[0], 1, device=self.device) - 0.5)
                pitch = default_euler[:,1] + 2.0 * goal_rotation_noise * (torch.rand(env_ids.shape[0], 1, device=self.device) - 0.5)
                yaw = default_euler[:,2] + 2.0 * goal_rotation_noise * (torch.rand(env_ids.shape[0], 1, device=self.device) - 0.5)
                quat = matrix_to_quaternion(euler_angles_to_matrix(torch.cat([roll, pitch, yaw], dim=-1), convention='XYZ'))
                self.ee_goal_buff[env_ids, 3:7] = quat           

                #     roll = -torch.pi + 2*torch.pi * torch.rand(
                #         size=(env_ids.shape[0],1), device=self.device) #roll from [-pi, pi)
                #     pitch = -torch.pi + 2*torch.pi * torch.rand(
                #         size=(env_ids.shape[0],1), device=self.device) #pitch from [-pi, pi)
                #     yaw = -torch.pi + 2*torch.pi * torch.rand(
                #         size=(env_ids.shape[0],1), device=self.device) #yaw from [-pi, pi)
                #     quat = matrix_to_quaternion(rpy_angles_to_matrix(torch.cat([roll, pitch, yaw], dim=-1)))
                #     curr_target_buff[env_ids, 3:7] = quat


            self.goal_ee_pos = self.ee_goal_buff[..., 0:3]
            self.goal_ee_quat = self.ee_goal_buff[..., 3:7]
            self.goal_ee_rot = quaternion_to_matrix(self.goal_ee_quat)

            self.prev_state_buff[env_ids] = torch.zeros_like(self.prev_state_buff[env_ids])
            goal_dict = dict(ee_goal=self.ee_goal_buff)
            reset_data['goal_dict'] = goal_dict

        return reset_data

    def update_params(self, param_dict):
        #Explicitly set parameters like goal states 
        goal_dict = param_dict['goal_dict']
        retract_state = goal_dict['retract_state'] if 'retract_state' in goal_dict else None
        ee_goal = goal_dict['ee_goal'] if 'ee_goal' in goal_dict else None
        joint_goal = goal_dict['joint_goal'] if 'joint_goal' in goal_dict else None

        if ee_goal is not None:
            goal_ee_pos = ee_goal[:, 0:3]
            goal_ee_quat = ee_goal[:, 3:7]
            goal_ee_rot = None        
        


        super(ArmReacher, self).update_params(retract_state=retract_state)
        
        if goal_ee_pos is not None:
            self.goal_ee_pos = torch.as_tensor(goal_ee_pos, device=self.device) # , dtype=self.dtype .unsqueeze(0)
            self.goal_state = None
        if goal_ee_rot is not None:
            self.goal_ee_rot = torch.as_tensor(goal_ee_rot, device=self.device) #.unsqueeze(0) dtype=self.dtype
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
            self.goal_state = None
        if goal_ee_quat is not None:
            self.goal_ee_quat = torch.as_tensor(goal_ee_quat, device=self.device)#.unsqueeze(0) , dtype=self.dtype
            self.goal_ee_rot = quaternion_to_matrix(self.goal_ee_quat)
            self.goal_state = None
        if joint_goal is not None:
            self.goal_state = torch.as_tensor(joint_goal, device=self.device)#.unsqueeze(0) , dtype=self.dtype
            self.goal_ee_pos, self.goal_ee_rot = self.dynamics_model.robot_model.compute_forward_kinematics(self.goal_state[:,0:self.n_dofs], self.goal_state[:,self.n_dofs:2*self.n_dofs], link_name=self.cfg['model']['ee_link_name'])
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
        
        return True
    
    @property
    def obs_dim(self)->int:
        return super().obs_dim + 7 #37 

    @property
    def action_lims(self)->Tuple[torch.Tensor, torch.Tensor]:
        act_highs = torch.tensor([self.cfg.max_acc] * self.action_dim,  device=self.device)
        act_lows = torch.tensor([-1.0 * self.cfg.max_acc] * self.action_dim, device=self.device)
        return act_lows, act_highs


