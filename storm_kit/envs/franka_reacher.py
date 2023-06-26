
import numpy as np
from storm_kit.envs.franka_env import FrankaEnv
from isaacgym import gymutil, gymtorch, gymapi

import torch

from storm_kit.mpc.cost import PoseCostQuaternion
from storm_kit.mpc.rollout import ArmReacher
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, matrix_to_quaternion, rpy_angles_to_matrix

class FrankaReacher(FrankaEnv):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        cfg['rollout']['num_instances'] = cfg['env']['numEnvs']
        cfg['rollout']['horizon'] = 1
        cfg['rollout']['num_particles'] = 1
        self.ee_link_name = cfg['rollout']['model']['ee_link_name']
        self.rollout_fn = ArmReacher(
            cfg['rollout'],
            world_params = cfg["world"],
            device=rl_device
        )
        self.state_dict = None

        num_obs = 41
        num_states = 22
        num_acts = 7

        cfg["env"]["numObservations"] = num_obs
        cfg["env"]["numStates"] = num_states
        cfg["env"]["numActions"] = num_acts

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        #TODO: Add world definition as well

        # pose_cost_params = self.cfg["env"]["cost"]["goal_pose"]

        # self.pose_cost = PoseCostQuaternion(
        #     **pose_cost_params,
        #     device = self.device,
        #     quat_inputs=True)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.goal_buf = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float
        )
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        self.target_poses = []
        for i in range(self.num_envs):
            target_pose_franka = torch.tensor([0.3, 0.0, 0.1, 0.0, 0.707, 0.707, 0.0], device=self.device, dtype=torch.float) 
            self.target_poses.append(target_pose_franka)
        self.target_poses = torch.cat(self.target_poses, dim=-1).view(self.num_envs, 7)

    def compute_reward(self):
        # st=time.time()
        # ee_pos = self.rigid_body_states[:, self.ee_handle][:, 0:3].unsqueeze(1).unsqueeze(1)
        # ee_quat = self.rigid_body_states[:, self.ee_handle][:, 3:7]
        # ee_quat = torch.cat((ee_quat[:,-1].unsqueeze(1), ee_quat[:,0:3]), dim=-1).unsqueeze(1).unsqueeze(1)
        # ee_quat = torch.roll(ee_quat, 1, -1).unsqueeze(1).unsqueeze(1)

        # target_pos = to_torch([self.target_pose_franka.p.x, self.target_pose_franka.p.y, self.target_pose_franka.p.z ])
        # target_rot = to_torch([self.target_pose_franka.r.w, self.target_pose_franka.r.x, self.target_pose_franka.r.y, self.target_pose_franka.r.z])
        # target_pos = target_pos.unsqueeze(0).expand(self.num_envs, target_pos.shape[0])
        # target_rot = target_rot.unsqueeze(0).expand(self.num_envs, target_rot.shape[0])
        
        # target_pos = self.target_poses[:, 0:3]
        # target_rot = self.target_poses[:, 3:7]

        # st1=time.time()
        # pose_cost, _, _ = self.pose_cost.forward(
        #     ee_pos, ee_quat, target_pos, target_rot
        # )
        # pose_cost = pose_cost.view(self.num_envs)

        # pose_reward = -1.0 * pose_cost
        # state_dict['ee_pos_seq'] = ee_pos
        # state_dict['ee_quat_seq'] = ee_quat

        # qpos =  self.franka_dof_pos
        # qvel =  self.franka_dof_vel
        # q_acc =  self.franka_dof_acc
        # ee_pos, ee_rot, lin_jac, ang_jac = self.robot_model.compute_fk_and_jacobian(qpos, qvel, link_name=self.ee_link_name)


        # state_dict['state_seq'] = torch.cat((qpos, qvel), dim=-1).unsqueeze(1).unsqueeze(1)
        # state_dict['ee_pos_seq'] = ee_pos
        # state_dict['ee_rot_seq'] = ee_rot
        # state_dict['lin_jac_seq'] = lin_jac
        # state_dict['ang_jac_seq'] = ang_jac

        # current_state = torch.cat((self.franka_dof_pos, self.franka_dof_vel, self.franka_dof_vel), dim=-1)
        cost, _ = self.rollout_fn.compute_cost(state_dict=self.state_dict, action_batch=self.actions, no_coll=True, horizon_cost=False)
        cost = cost.squeeze(1).squeeze(1)
        reward = -1.0 * cost #1.0 / (1.0 + cost) #-1.0 * cost

        self.rew_buf[:] = reward
        self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # self.extras['cost_terms'] = {
        #     'pose_cost': pose_cost
        # } 


    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        current_state = torch.cat((self.franka_dof_pos, self.franka_dof_vel, self.franka_dof_vel), dim=-1)

        obs, self.state_dict = self.rollout_fn.compute_observations(current_state=current_state)
        self.obs_buf[:] = obs.squeeze(1).squeeze(1)

        # ee_pos = self.rigid_body_states[:, self.ee_handle][:, 0:3]
        # ee_rot = self.rigid_body_states[:, self.ee_handle][:, 3:7]
        # ee_rot = torch.roll(ee_rot, 1, -1)

        # # target_pos = self.rigid_body_states[:, self.target_body_handle][:, 0:3]
        # # target_rot = self.rigid_body_states[:, self.target_body_handle][:, 3:7]
        # # target_base_rot = self.rigid_body_states[:, self.target_base_handle][:, 3:7]

        # # target_pos = to_torch([self.target_pose_franka.p.x, self.target_pose_franka.p.y, self.target_pose_franka.p.z])
        # # target_rot = to_torch([self.target_pose_franka.r.w, self.target_pose_franka.r.x, self.target_pose_franka.r.y, self.target_pose_franka.r.z])
        # # target_pos = target_pos.unsqueeze(0).expand(self.num_envs, target_pos.shape[0])
        # # target_rot = target_rot.unsqueeze(0).expand(self.num_envs, target_rot.shape[0])

        # # rot_err = torch.matmul(ee_rot, target_rot.T)
        # rot_product = ee_rot * target_rot
        # rot_err = torch.sum(rot_product, dim=-1)
        # rot_err = 1.0 - torch.abs(rot_err).unsqueeze(-1)

        # self.obs_buf[:] = torch.cat([
        #     self.franka_dof_pos, 
        #     self.franka_dof_vel,
        #     ee_pos, ee_rot,
        #     target_pos, target_rot,
        #     target_pos - ee_pos,
        #     # rot_product,
        #     rot_err
        # ], dim=-1)
        # #ee_rot, target_rot

        tstep = self.gym.get_sim_time(self.sim)
        self.states_buf[:] = torch.cat([
            self.franka_dof_pos,
            self.franka_dof_vel,
            self.franka_dof_acc,
            tstep*self.tstep
        ], dim=-1)

        target_pos = self.target_poses[:, 0:3]
        target_rot = self.target_poses[:, 3:7]

        #we return quaternion in wxyz format
        self.goal_buf[:] = torch.cat([
            target_pos,
            target_rot
        ], dim=-1)


    def post_physics_step(self):
        super().post_physics_step()
        self.extras["goal"] = self.goal_buf
        if self.viewer:
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                #plot target axes
                axes_geom = gymutil.AxesGeometry(0.1)
                # Create a wireframe sphere
                sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * np.pi, 0, 0)
                sphere_pose = gymapi.Transform(r=sphere_rot)
                sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(0, 1, 0))
                target_pos = self.target_poses[i, 0:3]
                target_rot = self.target_poses[i, 3:7]
                target_pos = gymapi.Vec3(x=target_pos[0], y=target_pos[1], z=target_pos[2]) 
                target_rot = gymapi.Quat(x=target_rot[1],y=target_rot[2], z=target_rot[3], w=target_rot[0])
                target_pose_franka = gymapi.Transform(p=target_pos, r=target_rot)
                target_pose_world = self.franka_pose_world * target_pose_franka
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], target_pose_world)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], target_pose_world)
                # #plot ee axes
                # ee_pos = self.rigid_body_states[i, self.ee_handle][0:3]
                # ee_rot = self.rigid_body_states[i, self.ee_handle][3:7]
                # ee_pos = gymapi.Vec3(x=ee_pos[0], y=ee_pos[1], z=ee_pos[2])
                # ee_rot = gymapi.Quat(x=ee_rot[0],y=ee_rot[1], z=ee_rot[2], w=ee_rot[3])
                # ee_pose_world = gymapi.Transform(p=ee_pos, r=ee_rot)
                # axes_geom = gymutil.AxesGeometry(0.1)
                # sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * np.pi, 0, 0)
                # sphere_pose = gymapi.Transform(r=sphere_rot)
                # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 1, 0))
                # gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], ee_pose_world)
                # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], ee_pose_world)
        # print('draw time', time.time()-st)


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        #reset EE target 
        if self.target_randomize_mode == "fixed_target":
            pass

        if self.target_randomize_mode in ["randomize_position", "randomize_full_pose"]:
            #randomize position
            self.target_poses[env_ids, 0] = 0.2 + 0.4 * torch.rand(
                size=(env_ids.shape[0],), device=self.device, dtype=torch.float32) #x from [0.2, 0.6)
            self.target_poses[env_ids, 1] = -0.3 + 0.6 * torch.rand(
                size=(env_ids.shape[0],), device=self.device, dtype=torch.float32) #y from [-0.3, 0.3)
            self.target_poses[env_ids, 2] = 0.2 + 0.3 * torch.rand(
                size=(env_ids.shape[0],), device=self.device, dtype=torch.float32) #z from [0.2, 0.5)
        
        if self.target_randomize_mode == "randomize_full_pose":
            #randomize orientation
            roll = -torch.pi + 2*torch.pi * torch.rand(
                size=(env_ids.shape[0],1), device=self.device, dtype=torch.float32) #roll from [-pi, pi)
            pitch = -torch.pi + 2*torch.pi * torch.rand(
                size=(env_ids.shape[0],1), device=self.device, dtype=torch.float32) #pitch from [-pi, pi)
            yaw = -torch.pi + 2*torch.pi * torch.rand(
                size=(env_ids.shape[0],1), device=self.device, dtype=torch.float32) #yaw from [-pi, pi)
            quat = matrix_to_quaternion(rpy_angles_to_matrix(torch.cat([roll, pitch, yaw], dim=-1)))
            self.target_poses[env_ids, 3:7] = quat
        
        #Update rollout function (task) with the new goals
        self.rollout_fn.update_params(self.target_poses)

    def reset(self):
        self.obs_dict = super().reset()
        self.obs_dict["goal"] = self.get_goal()

        return self.obs_dict

    def get_goal(self):
        return self.goal_buf.to(self.rl_device)
