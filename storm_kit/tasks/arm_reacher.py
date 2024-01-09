from typing import Dict, Optional, Tuple
import torch
from torch.profiler import record_function
from functools import reduce
from operator import mul

from storm_kit.differentiable_robot_model.spatial_vector_algebra import matrix_to_quaternion, quaternion_to_matrix, euler_angles_to_matrix, matrix_to_euler_angles, quat_multiply
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
        self.num_instances = 1
        
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

    def compute_observations(
            self, state_dict: Dict[str, torch.Tensor], 
            compute_full_state:bool = False, debug=False,
            cost_terms: Optional[Dict[str, torch.Tensor]]=None):

        if compute_full_state:
            state_dict = self._compute_full_state(state_dict, debug)
        
        obs =  super().compute_observations(state_dict, cost_terms)

        orig_size = state_dict['q_pos_seq'].size()[0:-1]
        # new_size = reduce(mul, list(orig_size))  
        # obs = obs.view(new_size, -1)

        ee_pos = state_dict['ee_pos_seq']
        ee_quat = state_dict['ee_quat_seq']
        ee_rot = state_dict['ee_rot_seq']
        

        if ee_pos.ndim == 2:
            ee_goal_pos = self.goal_ee_pos.expand_as(ee_pos)
            ee_goal_quat = self.goal_ee_quat.expand_as(ee_quat)
            ee_goal_rot = self.goal_ee_rot.reshape_as(ee_rot)
        elif ee_pos.ndim == 3:
            ee_goal_pos = self.goal_ee_pos.unsqueeze(1).expand_as(ee_pos)
            ee_goal_quat = self.goal_ee_quat.unsqueeze(1).expand_as(ee_quat)
            ee_goal_rot = self.goal_ee_rot.unsqueeze(1).reshape_as(ee_rot)
        elif ee_pos.ndim == 4:
            ee_goal_pos = self.goal_ee_pos.unsqueeze(1).unsqueeze(1).expand_as(ee_pos)
            ee_goal_quat = self.goal_ee_quat.unsqueeze(1).unsqueeze(1).expand_as(ee_quat)
            ee_goal_rot = self.goal_ee_rot.unsqueeze(1).unsqueeze(1).expand_as(ee_rot)

            # ee_goal_pos = ee_goal_pos.view(ee_goal_pos.shape[0], *(1,)*(len(target_shape)-ee_goal_pos.ndim), ee_goal_pos.shape[-1]).expand(target_shape)
            # ee_goal_quat = ee_goal_quat.view(ee_goal_quat.shape[0], *(1,)*(len(target_shape)-ee_goal_quat.ndim), ee_goal_quat.shape[-1]).expand(target_shape)
        
        if cost_terms is None:

            _, pose_cost_info = self.goal_cost.forward(
                ee_pos.view(-1, 3), ee_rot.view(-1,3,3), 
                ee_goal_pos.view(-1,3), ee_goal_rot.view(-1,3,3)
            )

            translation_res = pose_cost_info['translation_residual'].view(*orig_size,-1)
            rotation_res = pose_cost_info['rotation_residual'].view(*orig_size,-1)
        else:
            translation_res = cost_terms['translation_residual']
            rotation_res = cost_terms['rotation_residual']
        # ee_goal_pos, ee_goal_quat
        obs = torch.cat(
            (obs,
            translation_res, rotation_res), dim=-1)

        return obs

    def compute_cost(
            self, 
            state_dict: Dict[str, torch.Tensor], 
            action_batch: Optional[torch.Tensor]=None):

        cost, cost_terms = super(ArmReacher, self).compute_cost(
            state_dict = state_dict,
            action_batch = action_batch)

        q_pos_batch = state_dict['q_pos_seq']
        orig_size = q_pos_batch.size()[0:-1]
        new_size = reduce(mul, list(orig_size))  

        ee_pos, ee_rot = state_dict['ee_pos_seq'], state_dict['ee_rot_seq']

        goal_ee_pos = self.goal_ee_pos
        goal_ee_rot = self.goal_ee_rot
        goal_state = self.goal_state
        
        if goal_ee_pos is not None and goal_ee_rot is not None:
            if ee_pos.ndim == 2:
                goal_ee_pos = self.goal_ee_pos.reshape_as(ee_pos)
                goal_ee_rot = self.goal_ee_rot.reshape_as(ee_rot)
            elif ee_pos.ndim == 3:
                goal_ee_pos = self.goal_ee_pos.unsqueeze(1).reshape_as(ee_pos)
                goal_ee_rot = self.goal_ee_rot.unsqueeze(1).reshape_as(ee_rot)
            elif ee_pos.ndim == 4:
                goal_ee_pos = self.goal_ee_pos.unsqueeze(1).unsqueeze(1).repeat(1, ee_pos.shape[1], ee_pos.shape[2], 1)
                goal_ee_rot = self.goal_ee_rot.unsqueeze(1).unsqueeze(1).repeat(1, ee_rot.shape[1], ee_rot.shape[2], 1, 1)

            with record_function("pose_cost"):

                goal_cost, goal_cost_info = self.goal_cost.forward(
                    ee_pos.view(-1, 3), ee_rot.view(-1, 3, 3),
                    goal_ee_pos.view(-1, 3), goal_ee_rot.view(-1, 3, 3))
                
                goal_cost = goal_cost.view(orig_size)
                cost_terms['goal_pose_cost'] = goal_cost
                cost_terms['rotation_err'] = goal_cost_info['rotation_err'].view(orig_size)
                cost_terms['translation_err'] = goal_cost_info['translation_err'].view(orig_size)
                cost_terms['translation_residual'] = goal_cost_info['translation_residual'].view(*orig_size,-1)
                cost_terms['rotation_residual'] = goal_cost_info['rotation_residual'].view(*orig_size,-1)
                # goal_cost[:,:,0:-1] = 0.
                cost += goal_cost
        
        # joint l2 cost
        if self.cfg['cost']['joint_l2']['weight'] > 0.0 and goal_state is not None:
            disp_vec = q_pos_batch - goal_state[...,0:self.n_dofs]
            dist_cost = self.dist_cost.forward(disp_vec)
            cost += dist_cost.view(orig_size)
        
        return cost, cost_terms


    def compute_termination(self, 
            state_dict: Dict[str,torch.Tensor], 
            act_batch: Optional[torch.Tensor]=None,
            compute_full_state:bool = False):
        
        return super().compute_termination(
            state_dict, act_batch, compute_full_state)


    def compute_metrics(self, episode_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        q_pos = episode_data['states/q_pos'].to(self.device)
        q_vel = episode_data['states/q_vel'].to(self.device)
        q_acc = episode_data['states/q_acc'].to(self.device)
        state_dict = {'q_pos': q_pos, 'q_vel': q_vel, 'q_acc': q_acc}
        state_dict = self._compute_full_state(state_dict)

        ee_pos, ee_rot = state_dict['ee_pos_seq'], state_dict['ee_rot_seq']
        ee_quat = matrix_to_quaternion(ee_rot)

        ee_goal = episode_data['goal/ee_goal'].to(self.device)
        ee_goal_pos = ee_goal[:, 0:3]
        ee_goal_quat = ee_goal[:, 3:]

        # ee_goal_rot = quaternion_to_matrix(ee_goal_quat)
        # ee_pos, ee_rot, lin_jac_batch, ang_jac_batch = self.robot_model.compute_fk_and_jacobian(
        #     q_pos, self.cfg['ee_link_name'])
        
        conj_quat = ee_quat
        conj_quat[..., 1:] *= -1.0
        quat_res = quat_multiply(ee_goal_quat, conj_quat)

        dist_err = 100*torch.norm(ee_pos - ee_goal_pos, p=2, dim=-1) #l2 err in cm
        rot_err = 2.0 * torch.acos(quat_res[..., 0]) #rotation error in degrees
        #Last Step Error
        dist_err_final = dist_err[-1].item()
        rot_err_final = rot_err[-1].item()
        # dist_err_relative = final_goal_dist_err / dist_err[0].item()
        # rot_err_relative = final_goal_rot_err / dist_err[0].item()

        self.update_params(dict(goal_dict=dict(ee_goal=ee_goal)))
        cost, cost_terms = self.compute_cost(state_dict)
        
        #episode total cost
        total_cost = torch.sum(cost).item()

        #average manip score
        avg_manip_score = torch.mean(cost_terms['manip_score']).item()        
        
        #ee path length
        deltas = ee_pos[1:] - ee_pos[0:-1]
        deltas = torch.norm(deltas, p=2, dim=-1)
        ee_path_length = 100 * torch.sum(deltas).item() #ee path length in cm
    
        #max ee vel

        #termination
        term, term_cost, term_info = self.compute_termination(state_dict)
        world_collision_violation = term_info['in_coll_world'].sum(-1)
        self_collision_violation = term_info['in_coll_self'].sum(-1)
        bounds_violation = torch.logical_not(term_info['in_bounds']).sum(-1)

        #Last 10 step avg error
        # last_n_dist_err =  torch.mean(dist_err[-10:]).item()
        # last_n_dist_err_rel = last_n_dist_err / dist_err[0].item()
        
        #Max and average joint velocity
        # q_vel = episode_data['state_dict']['q_vel'].to(self.device)
        # q_vel_norm = torch.norm(q_vel, p=2, dim=-1)
        # max_q_vel = torch.max(q_vel_norm).item()
        # avg_q_vel = torch.mean(q_vel_norm).item()

        # #EE velocity
        # ee_vel_norm = torch.norm(ee_vel, p=2, dim=-1)
        # ee_ang_vel_norm = torch.norm(ee_ang_vel, p=2, dim=-1)
        # max_ee_vel = torch.max(ee_vel_norm).item()
        # max_ee_ang_vel = torch.max(ee_ang_vel_norm).item()
        # avg_ee_vel = torch.mean(ee_vel_norm).item()
        # avg_ee_ang_vel = torch.mean(ee_ang_vel_norm).item()

        return {
            'total_cost': total_cost,
            'dist_err_final': dist_err_final,
            'rot_err_final': rot_err_final,
            'avg_manip_score': avg_manip_score,
            'ee_path_length': ee_path_length,
            'world_collision': world_collision_violation.nonzero().numel(),
            'self_collision': self_collision_violation.nonzero().numel(),
            'bounds_violation': bounds_violation.nonzero().numel()}
            # 'last_10_dist_err': last_n_dist_err,
            # 'last_10_dist_err_rel': last_n_dist_err_rel,
            # 'max_q_vel': max_q_vel,
            # 'avg_q_vel': avg_q_vel,
            # 'max_ee_vel': max_ee_vel,
            # 'max_ee_ang_vel': max_ee_ang_vel,
            # 'avg_ee_vel': avg_ee_vel,
            # 'avg_ee_ang_vel': avg_ee_ang_vel}

    def reset(self, rng:Optional[torch.Generator]=None):
        env_ids = torch.arange(self.num_instances, device=self.device)
        return self.reset_idx(env_ids, rng=rng)

    def reset_idx(self, env_ids, rng:Optional[torch.Generator]=None):
        reset_data = {}
        if self.task_specs is not None:
            goal_position_noise = self.task_specs['target_position_noise']
            goal_rotation_noise = self.task_specs['target_rotation_noise']

            self.ee_goal_buff[env_ids] = self.default_ee_goal

            # if goal_position_noise > 0.:
            #randomize goal position around the default
            self.ee_goal_buff[env_ids, 0] = self.ee_goal_buff[env_ids, 0] +  goal_position_noise[0] * (2.0*torch.rand(self.ee_goal_buff[env_ids, 0].size(), device=self.device, generator=rng) - 1.0)
            self.ee_goal_buff[env_ids, 1] = self.ee_goal_buff[env_ids, 1] +  goal_position_noise[1] * (2.0*torch.rand(self.ee_goal_buff[env_ids, 1].size(), device=self.device, generator=rng) - 1.0)
            self.ee_goal_buff[env_ids, 2] = self.ee_goal_buff[env_ids, 2] +  goal_position_noise[2] * (2.0*torch.rand(self.ee_goal_buff[env_ids, 2].size(), device=self.device, generator=rng) - 1.0)
        
            # if goal_rotation_noise > 0.:
            #randomize goal orientation around defualt
            # TODO: fix
            # default_quat = self.ee_goal_buff[env_ids, 3:7]
            # default_euler = matrix_to_euler_angles(quaternion_to_matrix(default_quat), convention='XYZ')
            # roll = default_euler[:,0] + goal_rotation_noise[0] * (2.0*torch.rand(env_ids.shape[0], 1, device=self.device, generator=rng) - 1.0)
            # pitch = default_euler[:,1] + goal_rotation_noise[1] * (2.0*torch.rand(env_ids.shape[0], 1, device=self.device, generator=rng) - 1.0)
            # yaw = default_euler[:,2] + goal_rotation_noise[2] * (2.0*torch.rand(env_ids.shape[0], 1, device=self.device, generator=rng) - 1.0)
            # print(default_euler)
            # quat = matrix_to_quaternion(euler_angles_to_matrix(torch.cat([roll, pitch, yaw], dim=-1), convention='XYZ'))
            # self.ee_goal_buff[env_ids, 3:7] = quat 
            # print(self.ee_goal_buff)          

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

            # self.prev_state_buff[env_ids] = torch.zeros_like(self.prev_state_buff[env_ids])
            goal_dict = dict(ee_goal=self.ee_goal_buff)
            reset_data['goal_dict'] = goal_dict

        print(goal_dict)

        return reset_data

    def update_params(self, param_dict):
        #Explicitly set parameters like goal states 
        goal_dict = param_dict['goal_dict']
        retract_state = goal_dict['retract_state'] if 'retract_state' in goal_dict else None
        ee_goal = goal_dict['ee_goal'] if 'ee_goal' in goal_dict else None
        joint_goal = goal_dict['joint_goal'] if 'joint_goal' in goal_dict else None

        goal_ee_pos, goal_ee_quat, goal_ee_rot = None, None, None
        if ee_goal is not None:
            goal_ee_pos = ee_goal[:, 0:3]
            goal_ee_quat = ee_goal[:, 3:7]
            goal_ee_rot = None        
        
        super(ArmReacher, self).update_params(retract_state=retract_state)
        
        if goal_ee_pos is not None:
            self.goal_ee_pos = torch.as_tensor(goal_ee_pos, device=self.device)
            self.goal_state = None
        if goal_ee_rot is not None:
            self.goal_ee_rot = torch.as_tensor(goal_ee_rot, device=self.device) 
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
            self.goal_state = None
        if goal_ee_quat is not None:
            self.goal_ee_quat = torch.as_tensor(goal_ee_quat, device=self.device)
            self.goal_ee_rot = quaternion_to_matrix(self.goal_ee_quat)
            self.goal_state = None
        if joint_goal is not None:
            self.goal_state = torch.as_tensor(joint_goal, device=self.device)
            self.goal_ee_pos, self.goal_ee_rot = self.robot_model.compute_forward_kinematics(
                self.goal_state[:,0:self.n_dofs], link_name=self.cfg.model.ee_link_name)
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)

        return True
    
    @property
    def obs_dim(self)->int:
        return super().obs_dim + 6 

    @property
    def action_lims(self)->Tuple[torch.Tensor, torch.Tensor]:
        act_highs = torch.tensor([self.cfg.max_acc] * self.action_dim,  device=self.device)
        act_lows = torch.tensor([-1.0 * self.cfg.max_acc] * self.action_dim, device=self.device)
        return act_lows, act_highs


