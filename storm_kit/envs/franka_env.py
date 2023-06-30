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


class FrankaEnv(VecTask):

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
        self.world_params = self.cfg["world"]
        self.world_model = self.world_params["world_model"]
        
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

        franka_asset, franka_dof_props = self.load_franka_asset()
        table_asset, table_dims, table_color = self.load_table_asset()

        # temp = self.world_model["coll_objs"]["cube"]["table"]["pose"]
        table_pose_world = gymapi.Transform()
        table_pose_world.p = gymapi.Vec3(0, 0, 0 + table_dims.z)
        table_pose_world.r = gymapi.Quat(0., 0., 0., 1.)
        franka_start_pose_table = gymapi.Transform()
        franka_start_pose_table.p = gymapi.Vec3(-table_dims.x/2.0 + 0.2, 0.0, table_dims.z/2.0)
        franka_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.franka_pose_world =  table_pose_world * franka_start_pose_table #convert from franka to world frame
        
        trans = torch.tensor([
            self.franka_pose_world.p.x,
            self.franka_pose_world.p.y,
            self.franka_pose_world.p.z,
        ], device=self.rl_device).unsqueeze(0)
        quat = torch.tensor([
            self.franka_pose_world.r.w,
            self.franka_pose_world.r.x,
            self.franka_pose_world.r.y,
            self.franka_pose_world.r.z,
        ], device=self.rl_device).unsqueeze(0)
        rot = quaternion_to_matrix(quat)

        temp = CoordinateTransform(rot = rot, trans=trans)
        self.world_pose_franka = temp.inverse() #convert from world frame to franka

        # self.franka_pose_world = gymapi.Transform()
        # self.franka_pose_world.p = gymapi.Vec3(0.0, 0.0, 0.0)
        # self.franka_pose_world.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # self.target_pose_world = self.target_pose_franka * self.franka_pose_world
        # self.world_pose_franka = self.franka_pose_world.inverse() 

        # compute aggregate size
        max_agg_bodies = self.num_franka_bodies + 1 # #+ self.num_props * num_prop_bodies
        max_agg_shapes = self.num_franka_shapes + 1 #+ num_target_shapes #+ self.num_props * num_prop_shapes

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
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, self.franka_pose_world, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)            
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose_world, "table", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
    
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
        
        self.init_data()
    
    def init_data(self):
        
        self.ee_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "ee_link")


        #  get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "franka")

        self.root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.franka_jacobian = gymtorch.wrap_tensor(jacobian_tensor)


        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices
        # self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469], device=self.device)
        self.franka_default_dof_pos = to_torch([0.0, -0.7853, 0.0, -2.3561, 0.0, 1.5707, 0.7853], device=self.device)

        self.franka_dof_state = self.dof_state[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.franka_dof_acc = torch.zeros_like(self.franka_dof_vel)
        self.tstep = torch.ones(self.num_envs, 1, device=self.device)

        #TODO: Figure out if 13 is right
        self.num_bodies = self.rigid_body_states.shape[1]


        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device) 

        # self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.global_indices = torch.arange(self.num_envs * 1, dtype=torch.int32, device=self.device).view(self.num_envs, -1)


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
        
        # for env_ptr, franka_ptr, obj_ptr in zip(self.envs, self.frankas, self.objects):
        #     ee_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_ptr, "ee_link")
        #     ee_pose = self.gym.get_rigid_transform(env_ptr, ee_handle)
        #     obj_body_ptr = self.gym.get_actor_rigid_body_handle(env_ptr, obj_ptr, 0)
        #     self.gym.set_rigid_transform(env_ptr, obj_body_ptr, copy.deepcopy(ee_pose))

    def load_franka_asset(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../content/assets")
        franka_asset_file = "urdf/franka_description/franka_panda_no_gripper.urdf"
        # target_asset_file = "urdf/mug/movable_mug.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            # target_asset_file = self.cfg["env"]["asset"].get("assetFileNameTarget", target_asset_file)
        
        # self.robot_model = DifferentiableRobotModel(os.path.join(asset_root, franka_asset_file), None, device=self.device) #, dtype=self.dtype)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([40, 40, 40, 40, 40, 40, 40, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)

        # self.num_target_bodies = self.gym.get_asset_rigid_body_count(target_asset)
        # self.num_target_dofs = self.gym.get_asset_dof_count(target_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        # print("num target bodies:", self.num_target_bodies)
        # print("num target dofs:", self.num_target_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
        
        # set target object dof properties
        # target_dof_props = self.gym.get_asset_dof_properties(target_asset)
        # for i in range(self.num_target_dofs):
        #     target_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        #     target_dof_props['stiffness'][i] = 1000000.0
        #     target_dof_props['damping'][i] = 500.0
  

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

        return franka_asset, franka_dof_props


    def load_table_asset(self):
        #load table asset 
        table_dims = self.world_model["coll_objs"]["cube"]["table"]["dims"]
        table_dims=  gymapi.Vec3(table_dims[0], table_dims[1], table_dims[2])
        table_asset_options = gymapi.AssetOptions()
        # table_asset_options.armature = 0.001
        table_asset_options.fix_base_link = True
        # table_asset_options.thickness = 0.002
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z,
                                          table_asset_options)
        table_color = gymapi.Vec3(0.6, 0.6, 0.6)
        return table_asset, table_dims, table_color

    def world_to_franka(self, transform_world):
        transform_franka = self.world_pose_franka * transform_world
        return transform_franka

    def franka_to_world(self, transform_franka):
        transform_world = self.franka_pose_world * transform_franka
        return transform_world

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        self.actions = actions.clone().to(self.device)
        if self.control_space == "pos":
            targets = self.actions
        elif self.control_space == "vel":
            targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        elif self.control_space == "vel_2":
            targets = self.franka_dof_pos + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        elif self.control_space == "acc":
            raise NotImplementedError
    
        # targets = actions #self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)        
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))


    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.control_space == "pos":
            actions = actions['q_des'].clone().to(self.device)
        elif self.control_space == "vel":
            actions = actions['qd_des'].clone().to(self.device)
        elif self.control_space == "vel_2":
            actions = actions['qd_des'].clone().to(self.device)
        elif self.control_space == "acc":
            raise NotImplementedError
        return super().step(actions)


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
        pos = self.franka_default_dof_pos.unsqueeze(0)
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # multi_env_ids_int32 = self.global_indices[env_ids, 1:3].flatten()
        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))         
        # self.target_poses[env_ids, 0:3] = 0.2 + (0.6 - 0.2) * torch.rand(
        #      size=(env_ids.shape[0], 3), device=self.rl_device, dtype=torch.float)
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0 
        
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
        # self.obs_dict["goal"] = self.get_goal()

        return self.obs_dict


def compute_franka_reward():
    pass