from typing import Tuple, Dict, Any
from datetime import datetime
from os.path import join
import copy
import sys
import numpy as np
import os
import time


from isaacgym import gymutil, gymtorch, gymapi
import torch

from isaacgym.torch_utils import *
import yaml
from hydra.utils import instantiate

from storm_kit.differentiable_robot_model.spatial_vector_algebra import CoordinateTransform, quaternion_to_matrix
from storm_kit.envs.isaac_gym_env_utils import tensor_clamp, load_urdf_asset, load_primitive_asset
from storm_kit.envs.joint_controllers import JointStiffnessController, InverseDynamicsController


EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class IsaacGymRobotEnv():
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
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.num_environments = self.cfg["env"]["num_envs"]
        self.num_objects = self.cfg["env"]["num_objects"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.joint_control_mode = self.cfg["env"]["joint_control_mode"]
        self.robot_default_dof_pos = self.cfg["env"]["robot_default_dof_pos"]
        self.up_axis = "z"
        self.up_axis_idx = 2
        self.dt = self.cfg["sim"]["dt"]
        self.world_params = self.cfg["world"]
        self.world_model = self.world_params["world_model"]
        self.control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)                
        self.render_fps: int = self.cfg["env"].get("renderFPS", -1)
        self.robot_z_offset: float = self.cfg["env"].get("robot_z_offset", 0.0)
        self.floating_base_robot: bool = self.cfg["env"].get("floating_base_robot", False)
        self.ee_link_name = self.cfg.get("ee_link_name", "ee_link")
        
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

        # self.robot_dof_targets = None
        # self.effort_control = None

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
                # cam_pos = gymapi.Vec3(-5.0, -10.0, 3.0)
                # cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
                cam_pos = gymapi.Vec3(4.0, 0.0, 3.0)
                cam_target = gymapi.Vec3(-4.0, 0.0, 0.0)

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
        self.prev_action_buff = torch.zeros(
            (self.num_envs, self.num_robot_dofs), device=self.device)
        self.target_buf = torch.zeros(
            (self.num_envs, 7), device=self.device
        )
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

        #Load robot asset
        robot_asset_file = self.cfg["env"]["asset"].get("assetFileNameRobot")
        robot_asset = load_urdf_asset(
            self.gym, self.sim, asset_file=robot_asset_file, 
            fix_base_link=True, disable_gravity=True
        )

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)

        robot_dof_props = self.init_robot_dof_properties(robot_asset)

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        #Load a table
        table_color = gymapi.Vec3(0.6, 0.6, 0.6)
        table_dims = self.world_model["coll_objs"]["cube"]["table"]["dims"]
        table_dims=  gymapi.Vec3(table_dims[0], table_dims[1], table_dims[2])
        table_asset = load_primitive_asset(
            self.gym, self.sim, table_dims, asset_type='box', fix_base_link=True, disable_gravity=True)

        table_pose_world = gymapi.Transform()
        table_pose_world.p = gymapi.Vec3(0, 0, 0)
        table_pose_world.r = gymapi.Quat(0., 0., 0., 1.)

        #load objects
        self.num_object_bodies = 0
        self.num_object_shapes = 0
        if self.num_objects > 0:
            object_color = gymapi.Vec3(0.0, 0.0, 1.0)
            object_asset_file = self.cfg["env"]["asset"].get("assetFileNameObject")
            object_assets = []
            for _ in range(self.num_objects):
                object_asset = load_urdf_asset(
                    self.gym, self.sim, asset_file=object_asset_file, 
                    fix_base_link=True, disable_gravity=False)
                self.num_object_bodies += self.gym.get_asset_rigid_body_count(object_asset)
                self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
                object_assets.append(object_asset)
                object_start_pose_table = gymapi.Transform()
                object_start_pose_table.p = gymapi.Vec3(0.02 + 0.01 + 0.01, 0.0, table_dims.z/2.0 + 0.02) #0.3
                object_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            self.object_start_pose_world =  table_pose_world * object_start_pose_table #convert from franka to world frame

        robot_start_pose_table = gymapi.Transform()
        robot_start_pose_table.p = gymapi.Vec3(-table_dims.x/2.0 + 0.2, 0.0, 0.5 * table_dims.z + self.robot_z_offset)
        robot_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.robot_pose_world =  table_pose_world * robot_start_pose_table #convert from franka to world frame
        
        trans = torch.tensor([
            self.robot_pose_world.p.x,
            self.robot_pose_world.p.y,
            self.robot_pose_world.p.z,
        ], device=self.rl_device).unsqueeze(0)
        quat = torch.tensor([
            self.robot_pose_world.r.w,
            self.robot_pose_world.r.x,
            self.robot_pose_world.r.y,
            self.robot_pose_world.r.z,
        ], device=self.rl_device).unsqueeze(0)
        rot = quaternion_to_matrix(quat)

        temp = CoordinateTransform(rot = rot, trans=trans)
        self.world_pose_robot = temp.inverse() #convert from world frame to franka

        # compute aggregate size
        max_agg_bodies = self.num_robot_bodies + 1 + self.num_object_bodies # #+ self.num_props * num_prop_bodies
        max_agg_shapes = self.num_robot_shapes + 1 + self.num_object_shapes #+ num_target_shapes #+ self.num_props * num_prop_shapes

        self.tables = []
        self.robots = []
        self.envs = []
        self.objects = []

        for i in range(self.num_envs):
            env_objects = []
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, self.robot_pose_world, "robot", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            
            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)            
            
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose_world, "table", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.num_objects > 0:
                for i, object_asset in enumerate(object_assets):
                    object_actor = self.gym.create_actor(env_ptr, object_asset, self.object_start_pose_world, "ball_{}".format(i), i, 0, 0)
                    self.gym.set_rigid_body_color(env_ptr, object_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, object_color)
                    body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_actor)
                    for b in range(len(body_props)):
                        body_props[b].flags = gymapi.RIGID_BODY_NONE
                    self.gym.set_actor_rigid_body_properties(env_ptr, object_actor, body_props)
                    env_objects.append(object_actor)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.tables.append(table_actor)
            self.robots.append(robot_actor)
            self.objects.append(env_objects)
        
        self.init_data()
        self.init_joint_controller()


    def init_robot_dof_properties(self, robot_asset):
        robot_p_gains = self.cfg["env"].get("robot_p_gains")
        robot_d_gains = self.cfg["env"].get("robot_d_gains")
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_q_pos_lower_lims = []
        self.robot_q_pos_upper_lims = []
        self.robot_q_vel_lims = []
        self.robot_effort_lims = []
        for i in range(self.num_robot_dofs):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT #gymapi.DOF_MODE_POS

            if self.physics_engine == gymapi.SIM_PHYSX:
                robot_dof_props['stiffness'][i] = 0.0 #self.robot_dof_stiffness[i]
                robot_dof_props['damping'][i] = 0.0 # self.robot_dof_damping[i]
            else:
                robot_dof_props['stiffness'][i] = 7000.0
                robot_dof_props['damping'][i] = 50.0

            self.robot_q_pos_lower_lims.append(robot_dof_props['lower'][i])
            self.robot_q_pos_upper_lims.append(robot_dof_props['upper'][i])
            self.robot_q_vel_lims.append(robot_dof_props['velocity'][i])
            self.robot_effort_lims.append(robot_dof_props['effort'][i])
          

        self.robot_q_pos_lower_lims = torch.tensor(self.robot_q_pos_lower_lims, device=self.device).unsqueeze(0).repeat(self.num_envs,1)
        self.robot_q_pos_upper_lims = torch.tensor(self.robot_q_pos_upper_lims, device=self.device).unsqueeze(0).repeat(self.num_envs,1)
        self.robot_q_vel_lims = torch.tensor(self.robot_q_vel_lims, device=self.device).unsqueeze(0).repeat(self.num_envs,1)
        self.robot_effort_lims = torch.tensor(self.robot_effort_lims, device=self.device).unsqueeze(0).repeat(self.num_envs,1)
        self.robot_p_gains = torch.tensor(robot_p_gains, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.robot_d_gains = torch.tensor(robot_d_gains, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        # self.robot_dof_speed_scales = torch.ones_like(self.robot_q_pos_lower_lims)
        # self.robot_dof_stiffness = self.robot_dof_stiffness.unsqueeze(0).repeat(self.num_envs, 1)
        # self.robot_dof_damping = self.robot_dof_damping.unsqueeze(0).repeat(self.num_envs, 1)

        return robot_dof_props


    def init_data(self):
        
        self.ee_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], self.ee_link_name)

        #  get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        mass_tensor = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")

        self.root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.robot_jacobian = gymtorch.wrap_tensor(jacobian_tensor)
        self.robot_mass = gymtorch.wrap_tensor(mass_tensor)

        # create some wrapper tensors for different slices
        # self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469], device=self.device)
        # self.robot_default_dof_pos = to_torch([0.0, -0.7853, 0.0, -2.3561, 0.0, 1.5707, 0.7853], device=self.device)
        self.robot_default_dof_pos = to_torch(self.robot_default_dof_pos, device=self.device)
        self.robot_dof_state = self.dof_state[:, :self.num_robot_dofs]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]
        self.robot_dof_acc = torch.zeros_like(self.robot_dof_vel)
        self.episode_time = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_sim_time = torch.zeros(self.num_envs, 1, device=self.device)

        if self.floating_base_robot:
            self.robot_base_state = self.rigid_body_states[:, 3]
        self.ee_state = self.rigid_body_states[:, self.ee_handle]


        #Note: this needs to change to support more than one object!!!
        if self.num_objects > 0:
            self.object_state = self.root_state[:,-1]
            self.init_object_state = torch.zeros(self.num_envs, 13, device=self.device)
            self.init_object_state[:,0] = self.object_start_pose_world.p.x
            self.init_object_state[:,1] = self.object_start_pose_world.p.y
            self.init_object_state[:,2] = self.object_start_pose_world.p.z
            self.init_object_state[:,3] = self.object_start_pose_world.r.x
            self.init_object_state[:,4] = self.object_start_pose_world.r.y
            self.init_object_state[:,5] = self.object_start_pose_world.r.z
            self.init_object_state[:,6] = self.object_start_pose_world.r.w


        self.num_bodies = self.rigid_body_states.shape[1]
        self.target_poses = None

        # self._num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        # self.robot_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device) 
        self.effort_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device) 
        self.global_indices = torch.arange(self.num_envs * 1, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

    def init_joint_controller(self):
        if self.joint_control_mode == 'inverse_dynamics':
            self.joint_controller = InverseDynamicsController(
                p_gains = self.robot_p_gains,
                d_gains = self.robot_d_gains, device = self.device)
        elif self.joint_control_mode == 'joint_stiffness':
            self.joint_controller = JointStiffnessController(
                p_gains = self.robot_p_gains,
                d_gains = self.robot_d_gains, device = self.device)
        else:
            raise NotImplementedError('Joint control mode = {} not identified'.format(self.joint_control_mode))

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
    
    def pre_physics_step(self, actions:torch.Tensor):

        pos_des = actions[:, 0:self.num_robot_dofs].clone().to(self.device)
        vel_des = actions[:, self.num_robot_dofs:2*self.num_robot_dofs].clone().to(self.device)
        #feedforward componenet of desired acceleration
        acc_des = actions[:, 2*self.num_robot_dofs:3*self.num_robot_dofs].clone().to(self.device)

        if pos_des.ndim == 3:
            pos_des = pos_des[:, 0]
            vel_des = vel_des[:, 0]
            acc_des = acc_des[:, 0]

        # pos_des = tensor_clamp(pos_des, min=self.robot_q_pos_lower_lims, max=self.robot_q_pos_upper_lims)
        # vel_des = tensor_clamp(vel_des, min=-1.0 * self.robot_q_vel_lims, max=self.robot_q_vel_lims)

        curr_pos = self.robot_dof_pos
        curr_vel = self.robot_dof_vel

        if self.joint_control_mode == 'inverse_dynamics':
            torques = self.joint_controller.get_command(
                inertia_matrix = self.robot_mass, q_pos = curr_pos, q_vel = curr_vel, 
                q_pos_des = pos_des, q_vel_des = vel_des, q_acc_des = acc_des)
        elif self.joint_control_mode == 'joint_stiffness':
            torques = self.joint_controller.get_command(
                q_pos = curr_pos, q_vel = curr_vel, 
                q_pos_des = pos_des, q_vel_des = vel_des)

        torques = tensor_clamp(torques, min=-1.*self.robot_effort_lims, max=self.robot_effort_lims)
        self.effort_control[:,:] = torques

        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_control))
        self.prev_action_buff = acc_des.clone()


    def step(self, actions: torch.Tensor): # -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:

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
        
        return state_dict

    def get_state_dict(self):
        self._refresh()
        self.robot_q_pos_buff[:] = self.robot_dof_pos
        self.robot_q_vel_buff[:] = self.robot_dof_vel
        self.robot_q_acc_buff[:] = self.robot_dof_acc
        sim_time = self.gym.get_sim_time(self.sim)
        self.episode_time = sim_time - self.last_sim_time

        state_dict = {
            'q_pos': self.robot_q_pos_buff.to(self.rl_device),
            'q_vel': self.robot_q_vel_buff.to(self.rl_device),
            'q_acc': self.robot_q_acc_buff.to(self.rl_device),
            'prev_action': self.prev_action_buff.to(self.rl_device),
            'tstep': self.episode_time
        }
        
        if self.floating_base_robot:
            state_dict['base_pos'] = self.robot_base_state[:, 0:3].to(self.rl_device)
            state_dict['base_rot'] = self.robot_base_state[:, 3:7].to(self.rl_device)
            state_dict['base_vel'] = self.robot_base_state[:, 7:10].to(self.rl_device)

        if self.num_objects > 0:
            #Note: This won't work for more than one object
            state_dict['object_pos'] = self.object_state[:,0:3].to(self.rl_device)
            state_dict['object_rot'] = self.object_state[:,3:7].to(self.rl_device)
            state_dict['object_vel'] = self.object_state[:,7:10].to(self.rl_device)
            state_dict['object_ang_vel'] = self.object_state[:,10:13].to(self.rl_device)

        return state_dict
    
    def reset_idx(self, env_ids, reset_data=None):
        # reset franka
        # pos = tensor_clamp(
        #     self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
        #     self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        pos = self.robot_default_dof_pos.unsqueeze(0)
        self.robot_dof_pos[env_ids, :] = pos.clone()
        self.robot_dof_vel[env_ids, :] = torch.zeros_like(self.robot_dof_vel[env_ids])
        # self.robot_dof_targets[env_ids, :self.num_robot_dofs] = pos
        self.effort_control[env_ids, :] = torch.zeros_like(pos)

        # multi_env_ids_int32 = self.global_indices[env_ids, 1:3].flatten()
        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()
        multi_env_ids_object_int32 = self.global_indices[env_ids, -1].flatten()

        # self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self.robot_dof_targets),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))         

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0 
        self.prev_action_buff[env_ids] = torch.zeros_like(self.prev_action_buff[env_ids])
        if reset_data is not None:
            if 'goal_dict' in reset_data:
                self.update_goal(reset_data['goal_dict'])

        self.last_sim_time = torch.ones(self.num_envs, 1, device=self.device) * self.gym.get_sim_time(self.sim)
        state_dict = self.get_state_dict()

        return state_dict 


    def reset(self, reset_data=None):
        return self.reset_idx(torch.arange(self.num_envs, device=self.device), reset_data=reset_data)
        # state_dict = self.reset_idx(torch.arange(self.num_envs, device=self.device), reset_data=reset_data)
        # state_dict = self.get_state_dict()
        # return state_dict

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

            #draw targets etc.
            self.draw_auxillary_visuals()

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


    def draw_auxillary_visuals(self):
        #plot target or whatever else you want
        if self.viewer and self.target_buf is not None:
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                #plot target axes
                axes_geom = gymutil.AxesGeometry(0.1)
                # Plot sphere at target
                sphere_rot = gymapi.Quat.from_euler_zyx(0.0, 0, 0)
                sphere_pose = gymapi.Transform(r=sphere_rot)
                sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(0, 1, 0))

                target_pos = self.target_buf[i, 0:3]
                target_rot = self.target_buf[i, 3:7]
                target_pos = gymapi.Vec3(x=target_pos[0], y=target_pos[1], z=target_pos[2]) 
                target_rot = gymapi.Quat(x=target_rot[1],y=target_rot[2], z=target_rot[3], w=target_rot[0])
                if self.target_type == 'robot_goal':
                    target_pose_robot = gymapi.Transform(p=target_pos, r=target_rot)
                    target_pose_world = self.robot_pose_world * target_pose_robot
                elif self.target_type == 'object_goal':
                    target_pose_world = gymapi.Transform(p=target_pos, r=target_rot)
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], target_pose_world)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], target_pose_world)

                #plot ee axes
                axes_geom_ee = gymutil.AxesGeometry(0.1)
                # Plot sphere at target
                sphere_rot_ee = gymapi.Quat.from_euler_zyx(0.0, 0, 0)
                sphere_pose_ee = gymapi.Transform(r=sphere_rot_ee)
                sphere_geom_ee = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose_ee, color=(0, 0, 1))

                ee_pos = self.ee_state[i, 0:3]
                ee_rot = self.ee_state[i, 3:7]
                ee_pos = gymapi.Vec3(x=ee_pos[0], y=ee_pos[1], z=ee_pos[2]) 
                ee_rot = gymapi.Quat(x=ee_rot[0], y=ee_rot[1], z=ee_rot[2], w=ee_rot[3])
                ee_pose_world = gymapi.Transform(p=ee_pos, r=ee_rot)

                gymutil.draw_lines(axes_geom_ee, self.gym, self.viewer, self.envs[i], ee_pose_world)
                gymutil.draw_lines(sphere_geom_ee, self.gym, self.viewer, self.envs[i], ee_pose_world)


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
        if 'object_goal' in goal_dict:
            self.target_buf = goal_dict['object_goal']
            self.target_type = 'object_goal'
        else:
            self.target_buf = goal_dict['ee_goal']
            self.target_type = 'robot_goal'
    
    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_dofs(self):
        return self.num_robot_dofs

