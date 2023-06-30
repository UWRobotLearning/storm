
import numpy as np
from storm_kit.envs.franka_env import FrankaEnv
from isaacgym import gymutil, gymtorch, gymapi

import torch

from storm_kit.mpc.cost import PoseCostQuaternion
from storm_kit.mpc.rollout import ArmReacher
from storm_kit.differentiable_robot_model.coordinate_transform import CoordinateTransform, quaternion_to_matrix, matrix_to_quaternion, rpy_angles_to_matrix, transform_point

class FrankaPusher(FrankaEnv):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        cfg['rollout']['num_instances'] = cfg['env']['numEnvs']
        cfg['rollout']['horizon'] = 1
        cfg['rollout']['num_particles'] = 1
        self.ee_link_name = cfg['rollout']['model']['ee_link_name']
        self.rollout_fn = ArmReacher(
            cfg['rollout'],
            world_params=cfg['world'],
            device=rl_device
        )

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
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        franka_asset, franka_dof_props = self.load_franka_asset()
        table_asset, table_dims, table_color = self.load_table_asset()
        ball_asset, ball_color = self.load_ball_asset()

        temp = self.world_model["coll_objs"]["cube"]["table"]["pose"]
        table_pose_world = gymapi.Transform()
        table_pose_world.p = gymapi.Vec3(temp[0], temp[1], temp[2] + table_dims.z)
        table_pose_world.r = gymapi.Quat(0., 0., 0., 1.)
        franka_start_pose_table = gymapi.Transform()
        franka_start_pose_table.p = gymapi.Vec3(-table_dims.x/2.0 + 0.2, 0.0, table_dims.z/2.0)
        franka_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.franka_pose_world =  table_pose_world * franka_start_pose_table #convert from franka to world frame
        
        ball_start_pose_table = gymapi.Transform()
        ball_start_pose_table.p = gymapi.Vec3(-table_dims.x/2.0 + 0.5, 0.0, table_dims.z/2.0)
        ball_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.ball_start_pose_world =  table_pose_world * ball_start_pose_table #convert from franka to world frame

        trans = torch.tensor([
            self.franka_pose_world.p.x,
            self.franka_pose_world.p.y,
            self.franka_pose_world.p.z,
        ], device=self.device).unsqueeze(0)
        quat = torch.tensor([
            self.franka_pose_world.r.w,
            self.franka_pose_world.r.x,
            self.franka_pose_world.r.y,
            self.franka_pose_world.r.z,
        ], device=self.device).unsqueeze(0)
        rot = quaternion_to_matrix(quat)

        temp = CoordinateTransform(rot = rot, trans=trans, device=self.device)
        self.world_pose_franka = temp.inverse() #convert from world frame to franka


        # self.franka_pose_world = gymapi.Transform()
        # self.franka_pose_world.p = gymapi.Vec3(0.0, 0.0, 0.0)
        # self.franka_pose_world.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # self.target_pose_world = self.target_pose_franka * self.franka_pose_world
        # self.world_pose_franka = self.franka_pose_world.inverse() #convert from world to franka

        # compute aggregate size
        max_agg_bodies = self.num_franka_bodies + 1 + 1 # franka + table + ball
        max_agg_shapes = self.num_franka_shapes + 1 + 1 # franka + table + ball

        # self.tables = []
        self.frankas = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, self.franka_pose_world, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)            
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose_world, "table", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self.ball_actor = self.gym.create_actor(env_ptr, ball_asset, self.ball_start_pose_world, "ball", i, 2, 0)
            self.gym.set_rigid_body_color(env_ptr, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, ball_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
        
        self.ee_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "ee_link")
        self.init_data()


    # def init_data(self):
    #     super().init_data()
    #     self.ball_body_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ball_actor, "box")
    #     self.ball_state = self.root_state[:,self.ball_actor,:]
    #     self.ball_pos = self.ball_state[:, 0:3]
    #     # self.ball_pose_franka =   #convert from franka to world frame
    #     # print(self.ball_pos, self.ball_pos.device, self.world_pose_franka.device, self.device, self.rl_device)
    #     # print(self.world_pose_franka.transform_point(self.ball_pos))
    #     # exit()


    #     target_pose_franka = torch.tensor([0.5, 0.0, 0.5, 0.0, 0.707, 0.707, 0.0], device=self.device, dtype=torch.float) 
    #     self.target_poses = target_pose_franka.unsqueeze(0).repeat(self.num_envs,1)

    def load_ball_asset(self):
        #load table asset 
        ball_radius =  0.03
        ball_asset_options = gymapi.AssetOptions()
        # ball_asset_options.armature = 0.001
        # table_asset_options.fix_base_link = True
        # table_asset_options.thickness = 0.002
        ball_asset = self.gym.create_sphere(self.sim, ball_radius, ball_asset_options)
        ball_color = gymapi.Vec3(0.0, 0.0, 1.0)
        return ball_asset, ball_color
    

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
        # print(cost, reward)


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
        self._refresh()

        self.ball_body_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ball_actor, "box")
        self.ball_state = self.root_state[:,self.ball_actor,:]
        ball_pos_world = self.ball_state[:, 0:3]
        ball_pos_franka = self.world_pose_franka.transform_point(ball_pos_world).squeeze(1)

        ball_quat_franka = torch.tensor([1.0, 0., 0., 0.0], device=self.device, dtype=torch.float).unsqueeze(0).repeat(self.num_envs,1)
        
        self.target_poses = torch.cat((ball_pos_franka, ball_quat_franka), dim=-1)

        #reset EE target 
        if self.target_randomize_mode == "fixed_target":
            pass

        if self.target_randomize_mode == "randomize_position":
            #randomize position
            self.target_poses[env_ids, 0] = 0.2 + 0.4 * torch.rand(
                size=(env_ids.shape[0],), device=self.device, dtype=torch.float32) #x from [0.2, 0.6)
            self.target_poses[env_ids, 1] = -0.3 + 0.6 * torch.rand(
                size=(env_ids.shape[0],), device=self.device, dtype=torch.float32) #y from [-0.3, 0.3)
            self.target_poses[env_ids, 2] = 0.2 + 0.3 * torch.rand(
                size=(env_ids.shape[0],), device=self.device, dtype=torch.float32) #z from [0.2, 0.5)
                
        #Update rollout function (task) with the new goals
        self.rollout_fn.update_params(self.target_poses)

    def reset(self):
        self.obs_dict = super().reset()
        self.obs_dict["goal"] = self.get_goal()

        return self.obs_dict

    def get_goal(self):
        return self.goal_buf.to(self.rl_device)
