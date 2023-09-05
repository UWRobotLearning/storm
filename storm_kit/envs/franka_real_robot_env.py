from typing import Tuple, Dict, Any
from datetime import datetime
from os.path import join
import copy
import sys
import numpy as np
import os
import time

import torch

import yaml

from storm_kit.differentiable_robot_model.coordinate_transform import CoordinateTransform, quaternion_to_matrix, matrix_to_quaternion, rpy_angles_to_matrix


EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class FrankaEnv(): #VecTask
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):        
        self.cfg = cfg

        self.rl_device = rl_device
        self.device = sim_device
        device_name = torch.cuda.get_device_name(sim_device)
        split_device = device_name.split(':')
        self.device_id = split_device[1] if len(split_device) > 1 else 0 

        if self.cfg["sim"]["use_gpu_pipeline"]:
            if self.device == torch.device('cpu'):
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                self.cfg["sim"]["use_gpu_pipeline"] = False

        self.headless = headless
        enable_camera_sensors = self.cfg.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()


        self.force_render = force_render
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.control_space = self.cfg["env"]["controlSpace"]
        self.num_environments = self.cfg["env"]["num_envs"]
        # self.num_objects = self.cfg["env"]["num_objects"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.up_axis = "z"
        self.up_axis_idx = 2
        self.dt = self.cfg["sim"]["dt"]
        self.world_params = self.cfg["world"]
        self.world_model = self.world_params["world_model"]


        self.control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
                
        self.render_fps: int = self.cfg["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0

        self.total_train_env_frames: int = 0

        # number of control steps
        self.control_steps: int = 0

        self.record_frames: bool = False
        self.record_frames_dir = join("recorded_frames", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        
        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        self.dt: float = self.sim_params.dt

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        # self.first_randomization = True
        # self.original_props = {}
        # self.dr_randomizations = {}
        # self.actor_params_generator = None
        # self.extern_actor_params = {}
        # self.last_step = -1
        # self.last_rand_step = -1
        # for env_id in range(self.num_envs):
        #     self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()
        
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

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        # self.obs_buf = torch.zeros(
        #     (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        # self.states_buf = torch.zeros(
        #     (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        # self.rew_buf = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.float)
        self.robot_q_pos_buff = torch.zeros(
            (self.num_envs, self.num_robot_dofs), device=self.device)
        self.robot_q_vel_buff = torch.zeros(
            (self.num_envs, self.num_robot_dofs), device=self.device)
        self.robot_q_acc_buff = torch.zeros(
            (self.num_envs, self.num_robot_dofs), device=self.device)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        # self.randomize_buf = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

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
        # self.sim = super().create_sim(
        #     self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.sim = _create_sim_once(self.gym, self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

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
        self.target_poses = None

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

    def load_object_asset(self, disable_gravity=False):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../content/assets")
        object_asset_file = "urdf/ball.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            object_asset_file = self.cfg["env"]["asset"].get("assetFileNameObject", object_asset_file )
        
        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = disable_gravity
        asset_options.thickness = 0.001
        asset_options.use_mesh_materials = True
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)

        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        print("num object bodies: ", self.num_object_bodies)
        object_color = gymapi.Vec3(0.0, 0.0, 1.0)

        return object_asset, object_color 


    def world_to_franka(self, transform_world):
        transform_franka = self.world_pose_franka * transform_world
        return transform_franka

    def franka_to_world(self, transform_franka):
        transform_world = self.franka_pose_world * transform_franka
        return transform_world

    def pre_physics_step(self, action_dict: Dict[str, torch.Tensor]):
        # implement pre-physics simulation code here
        #    - e.g. apply actions

        if self.control_space == "pos":
            actions = action_dict['q_pos_des'].clone().to(self.device)
            targets = actions
        elif self.control_space == "vel":
            actions = action_dict['q_vel_des'] if 'q_vel_des' in action_dict else action_dict['raw_action']
            actions = actions.clone().to(self.device)
            targets = self.franka_dof_targets[:, :self.num_franka_dofs] + \
                  self.franka_dof_speed_scales * self.dt * actions * self.action_scale

        elif self.control_space == "vel_2":
            actions = action_dict['q_vel_des'].clone().to(self.device)
            targets = self.franka_dof_pos + \
                self.franka_dof_speed_scales * self.dt * actions * self.action_scale

        elif self.control_space == "acc":
            raise NotImplementedError
    
        # targets = actions #self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)        
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))


    def step(self, actions: Dict[str, torch.Tensor]): # -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:

        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """



        # randomize actions
        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # action_tensor = actions
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == torch.device('cpu'):
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        state_dict = self.post_physics_step()

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # # randomize observations
        # if self.dr_randomizations.get('observations', None):
        #     self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        # self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        # self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # # asymmetric actor-critic
        # if self.num_states > 0:
        #     self.obs_dict["states"] = self.get_state()

        # return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras


        return state_dict, self.reset_buf.to(self.rl_device)


    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1                
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        # self.compute_observations()
        # self.compute_reward()
        state_dict = self.get_state_dict()
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        if self.viewer and self.target_poses is not None:
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

        return state_dict

    def get_state_dict(self):
        self._refresh()
        self.robot_q_pos_buff[:] = self.franka_dof_pos
        self.robot_q_vel_buff[:] = self.franka_dof_vel
        self.robot_q_acc_buff[:] = self.franka_dof_acc
        tstep = self.gym.get_sim_time(self.sim)
        tstep *= self.tstep


        state_dict = {
            'q_pos': self.robot_q_pos_buff.to(self.rl_device),
            'q_vel': self.robot_q_vel_buff.to(self.rl_device),
            'q_acc': self.robot_q_acc_buff.to(self.rl_device),
            'tstep': tstep
        }
        return state_dict
    
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

        state_dict = self.get_state_dict()
        return state_dict 


    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # self.compute_observations()
        state_dict = self.get_state_dict()

        # self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        # if self.num_states > 0:
        #     self.obs_dict["states"] = self.get_state()
        # self.obs_dict["goal"] = self.get_goal()

        return state_dict #self.obs_dict


    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.dt * self.control_freq_inv  # render every control step
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"frame_{self.control_steps}.png"))

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)


    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params


    def update_goal(self, goal_dict):
        self.target_poses = goal_dict['ee_goal']

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    # @property
    # def num_acts(self) -> int:
    #     """Get the number of actions in the environment."""
    #     return self.num_actions

    @property
    def num_robot_dofs(self):
        return self.num_franka_dofs