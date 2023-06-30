from typing import Tuple, Dict, Any
import copy
import numpy as np
import os
import time


from isaacgym import gymutil, gymtorch, gymapi
import torch

from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
import yaml
from hydra.utils import instantiate

# from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.coordinate_transform import CoordinateTransform, quaternion_to_matrix, matrix_to_quaternion, rpy_angles_to_matrix


class SimplePusher(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.control_space = self.cfg["env"]["controlSpace"]
        self.target_randomize_mode = self.cfg["env"]["targetRandomizeMode"]
    
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.dt = self.cfg["sim"]["dt"]

        cfg["env"]["numObservations"] = 5
        cfg["env"]["numStates"] = 5
        cfg["env"]["numActions"] = 2
        
        super().__init__(
            config=self.cfg, 
            rl_device=rl_device, 
            sim_device=sim_device, 
            graphics_device_id=graphics_device_id, 
            headless=headless, 
            virtual_screen_capture=virtual_screen_capture, 
            force_render=force_render)
    
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(-5.0, -10.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        #    - call super().create_sim with device args (see docstring)
        #    - create ground plane
        #    - set up environments
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        table_asset, table_dims, table_color = self.load_table_asset()
        ball_asset, ball_color = self.load_ball_asset()
        robot_asset, robot_color = self.load_point_robot_asset()


        table_pose_world = gymapi.Transform()
        table_pose_world.p = gymapi.Vec3(0, 0, 0 + table_dims.z)
        table_pose_world.r = gymapi.Quat(0., 0., 0., 1.)

        ball_start_pose_table = gymapi.Transform()
        ball_start_pose_table.p = gymapi.Vec3(-table_dims.x/2.0 + 0.5, 0.0, table_dims.z/2.0)
        ball_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.ball_start_pose_world =  table_pose_world * ball_start_pose_table #convert from franka to world frame

        robot_start_pose_table = gymapi.Transform()
        robot_start_pose_table.p = gymapi.Vec3(-table_dims.x/2.0 - 0.1 + 0.5, 0.0, table_dims.z/2.0)
        robot_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.robot_start_pose_world =  table_pose_world * robot_start_pose_table #convert from franka to world frame


        max_agg_bodies = 1 + 1 + 1 # robot + table + ball
        max_agg_shapes = 1 + 1 + 1 # robot + table + ball

        # self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_bodies = 3
        self.envs = []
        self.robots = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            robot_actor = self.gym.create_actor(env_ptr, robot_asset, self.robot_start_pose_world, "robot", i, 0, 0)

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
            self.robots.append(robot_actor)
            # self.frankas.append(franka_actor)
        
        # self.ee_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "ee_link")
        self.init_data()

    def _update_states(self):
        pass

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()


    def init_data(self):
        pass

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
        

    def load_table_asset(self):
        #load table asset 
        # table_dims = self.world_model["coll_objs"]["cube"]["table"]["dims"]
        
        table_dims=  gymapi.Vec3(1, 1, 0.1)
        table_asset_options = gymapi.AssetOptions()
        # table_asset_options.armature = 0.001
        table_asset_options.fix_base_link = True
        # table_asset_options.thickness = 0.002
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z,
                                          table_asset_options)
        table_color = gymapi.Vec3(0.6, 0.6, 0.6)
        return table_asset, table_dims, table_color

    def load_point_robot_asset(self):
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = False
        robot_asset_options.disable_gravity = True
        robot_asset = self.gym.create_sphere(self.sim, 0.01, robot_asset_options)
        robot_color = gymapi.Vec3(1.0, 0.0, 0.0)
        return robot_asset, robot_color
 
  

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        self.actions = actions.clone().to(self.device)
        forces = torch.zeros((self.num_envs, self.num_robot_bodies, 3), device=self.device, dtype=torch.float)
        torques = torch.zeros((self.num_envs, self.num_robot_bodies, 3), device=self.device, dtype=torch.float)
        forces[:, 0, 2] = 300
        # torques[:, 0, 2] = torque_amt
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

    
        # targets = actions #self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        # self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
        #     targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # # env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)        
        # self.gym.set_dof_position_target_tensor(self.sim,
        #                                         gymtorch.unwrap_tensor(self.franka_dof_targets))
    

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        self.compute_reward()


    def reset_idx(self, env_ids):

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset franka
        # pos = tensor_clamp(
        #     self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
        #     self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # pos = self.franka_default_dof_pos.unsqueeze(0)
        # self.franka_dof_pos[env_ids, :] = pos
        # self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        # self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # # multi_env_ids_int32 = self.global_indices[env_ids, 1:3].flatten()
        # multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()
        # self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self.franka_dof_targets),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))         
        # self.target_poses[env_ids, 0:3] = 0.2 + (0.6 - 0.2) * torch.rand(
        #      size=(env_ids.shape[0], 3), device=self.rl_device, dtype=torch.float)
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0 
    

    def compute_observations(self):
        pass

    def compute_reward(self):
        pass