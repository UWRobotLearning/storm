import copy
import os
from typing import Dict, Any, Tuple, List, Set

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch
from isaacgymenvs.utils.dr_utils import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import torch
import numpy as np
import operator, random
from copy import deepcopy
from isaacgymenvs.utils.utils import nested_dict_get_attr, nested_dict_set_attr

from collections import deque

import sys

import abc
from abc import ABC

from isaacgymenvs.tasks.base.vec_task import Env

# import numpy as np
# import os 

# from isaacgym import gymtorch, gymapi, gymutil
# import torch
# import torch.nn as nn

# from isaacgym.torch_utils import *


from storm_kit.util_file import get_assets_path

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class VecTask(Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False): 
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
            force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)
        self.cfg = config
        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
        self.force_render = force_render

        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        # self.sim = self.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # self._create_ground_plane()
        # self._create_envs(self.num_envs, 1.5, 1)
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()

        self.obs_dict = {}

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
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
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
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    # def _create_ground_plane(self):
    #     plane_params = gymapi.PlaneParams()
    #     plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    #     self.gym.add_ground(self.sim, plane_params)


    # def _create_envs(self, num_envs, spacing, num_per_row):
    #     lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    #     upper = gymapi.Vec3(spacing, spacing, spacing)

    #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../content/assets")
    #     franka_asset_file = "urdf/cartpole.urdf"
    #     target_asset_file = "urdf/mug/movable_mug.urdf"

    #     if "asset" in self.cfg["env"]:
    #         asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
    #         franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
    #         target_asset_file = self.cfg["env"]["asset"].get("assetFileNameTarget", target_asset_file)
    #     #load table asset 
    #     # table_dims = self.world_model["coll_objs"]["cube"]["table"]["dims"]
    #     # table_dims=  gymapi.Vec3(table_dims[0], table_dims[1], table_dims[2])
    #     # table_asset_options = gymapi.AssetOptions()
    #     # table_asset_options.armature = 0.001
    #     # table_asset_options.fix_base_link = True
    #     # table_asset_options.thickness = 0.002
    #     # table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z,
    #     #                                   table_asset_options)
    #     # table_color = gymapi.Vec3(0.6, 0.6, 0.6)


    #     # load franka asset
    #     asset_options = gymapi.AssetOptions()
    #     asset_options.flip_visual_attachments = True
    #     asset_options.fix_base_link = True
    #     asset_options.collapse_fixed_joints = False
    #     asset_options.disable_gravity = True
    #     asset_options.thickness = 0.001
    #     asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    #     asset_options.use_mesh_materials = True
    #     franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

    #     # #load target mug asset
    #     # target_asset_options = gymapi.AssetOptions()
    #     # target_asset_options.flip_visual_attachments = False
    #     # target_asset_options.fix_base_link = True
    #     # target_asset_options.collapse_fixed_joints = False
    #     # target_asset_options.disable_gravity = True
    #     # target_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    #     # target_asset_options.armature = 0.001
    #     # target_asset_options.use_mesh_materials = True
    #     # target_asset = self.gym.load_asset(self.sim, asset_root, target_asset_file, target_asset_options)

    #     franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
    #     franka_dof_damping = to_torch([40, 40, 40, 40, 40, 40, 40, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

    #     self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    #     self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
    #     # self.num_target_bodies = self.gym.get_asset_rigid_body_count(target_asset)
    #     # self.num_target_dofs = self.gym.get_asset_dof_count(target_asset)

    #     print("num franka bodies: ", self.num_franka_bodies)
    #     print("num franka dofs: ", self.num_franka_dofs)
    #     # print("num target bodies:", self.num_target_bodies)
    #     # print("num target dofs:", self.num_target_dofs)

    #     # set franka dof properties
    #     franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
    #     self.franka_dof_lower_limits = []
    #     self.franka_dof_upper_limits = []
    #     for i in range(self.num_franka_dofs):
    #         franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    #         if self.physics_engine == gymapi.SIM_PHYSX:
    #             franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
    #             franka_dof_props['damping'][i] = franka_dof_damping[i]
    #         else:
    #             franka_dof_props['stiffness'][i] = 7000.0
    #             franka_dof_props['damping'][i] = 50.0

    #         self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
    #         self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
        
    #     # set target object dof properties
    #     # target_dof_props = self.gym.get_asset_dof_properties(target_asset)
    #     # for i in range(self.num_target_dofs):
    #     #     target_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    #     #     target_dof_props['stiffness'][i] = 1000000.0
    #     #     target_dof_props['damping'][i] = 500.0
  

    #     self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
    #     self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
    #     self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

    #     # temp = self.world_model["coll_objs"]["cube"]["table"]["pose"]
    #     # table_pose_world = gymapi.Transform()
    #     # table_pose_world.p = gymapi.Vec3(temp[0], temp[1], temp[2] + table_dims.z)
    #     # table_pose_world.r = gymapi.Quat(temp[3], temp[4], temp[5], temp[6])



    #     # franka_start_pose_table = gymapi.Transform()
    #     # franka_start_pose_table.p = gymapi.Vec3(-table_dims.x/2.0, 0.0, table_dims.z/2.0)
    #     # franka_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    #     # self.target_pose_franka = gymapi.Transform()
    #     # self.target_pose_franka.p = gymapi.Vec3(0.5, 0.0, 0.5)
    #     # self.target_pose_franka.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    #     # self.franka_pose_world =  table_pose_world * franka_start_pose_table #convert from franka to world frame
        
    #     self.franka_pose_world = gymapi.Transform()
    #     self.franka_pose_world.p = gymapi.Vec3(0.0, 0.0, 0.0)
    #     self.franka_pose_world.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    #     # self.target_pose_world = self.target_pose_franka * self.franka_pose_world
    #     # self.world_pose_franka = self.franka_pose_world.inverse() #convert from world to franka

    #     # compute aggregate size
    #     # num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    #     # num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    #     # num_obj_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
    #     # num_target_shapes = self.gym.get_asset_rigid_shape_count(target_asset)
    #     # max_agg_bodies = self.num_franka_bodies # + self.num_target_bodies #+ self.num_props * num_prop_bodies
        
    #     # max_agg_shapes = num_franka_shapes #+ num_target_shapes #+ self.num_props * num_prop_shapes

    #     # self.tables = []
    #     self.frankas = []
    #     # self.targets = []
    #     self.envs = []
    #     self.target_poses = []

    #     for i in range(self.num_envs):
    #         # create env instance
    #         env_ptr = self.gym.create_env(
    #             self.sim, lower, upper, num_per_row
    #         )

    #         # if self.aggregate_mode >= 3:
    #         # self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
    #         # table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose_world, "table", i, 1, 0)
    #         # self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

    #         franka_actor = self.gym.create_actor(env_ptr, franka_asset, self.franka_pose_world, "franka", i, 1, 0)
    #         self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
    #         # target_actor = self.gym.create_actor(env_ptr, target_asset, self.target_pose_world, "target", i, 1, 0)
    #         target_pose_franka = torch.tensor([0.5, 0.0, 0.5, 0.0, 0.707, 0.707, 0.0], device=self.device, dtype=torch.float) 
    #         self.target_poses.append(target_pose_franka)
    #         # if self.aggregate_mode == 2:
    #         #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)            

    #         # if self.aggregate_mode == 1:
    #         #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
    
    #         # if self.aggregate_mode > 0:
    #         # self.gym.end_aggregate(env_ptr)

    #         self.envs.append(env_ptr)
    #         # self.tables.append(table_actor)
    #         self.frankas.append(franka_actor)
    #         # self.targets.append(target_actor)
        
    #     self.ee_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "ee_link")

    #     self.target_poses = torch.cat(self.target_poses, dim=-1).view(self.num_envs, 7)

    #     # ee_pose = self.gym.get_rigid_transform(env_ptr, self.ee_handle)
    #     # self.target_pose_world.r = ee_pose.r
    #     # self.target_pose_franka = self.franka_pose_world.inverse() * self.target_pose_world



    #     # self.ee_pose = self.gym.get_rigid_transform(env_ptr, self.ee_handle)

    #     # self.target_base_handle = self.gym.find_actor_rigid_body_handle(env_ptr, target_actor, "base")
    #     # self.target_body_handle = self.gym.find_actor_rigid_body_handle(env_ptr, target_actor, "mug")

    #     # self.init_data()
    

    def get_state(self):
        """Returns the state buffer of the environment (the privileged observations for asymmetric training)."""
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    # @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """
        pass

    # @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
        pass

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)

        return actions

    def reset_idx(self, env_idx):
        """Reset environment with indces in env_idx. 
        Should be implemented in an environment class inherited from VecTask.
        """  
        pass

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_done(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

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

            else:
                self.gym.poll_viewer_events(self.viewer)

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

    """
    Domain Randomization methods
    """

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name+'_' + str(prop_idx) + '_'+attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    def apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over actors, then loop over envs, then loop over their props 
        # and lastly loop over the ranges of the params 

        for actor, actor_properties in dr_params["actor_params"].items():

            # Loop over all envs as this part is not tensorised yet 
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                # randomise dof_props, rigid_body, rigid_shape properties 
                # all obtained from the YAML file
                # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties  
                #          prop_attrs: 
                #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue

                    if prop_name == 'scale':
                        setup_only = prop_attrs.get('setup_only', False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1,
                                                             self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params['operation'] == 'scaling':
                                new_scale = og_scale * sample
                            elif attr_randomization_params['operation'] == 'additive':
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True

                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get('setup_only', False)
                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                            extern_sample, extern_offsets[env_id], p, attr)
                                    apply_random_samples(
                                        p, og_p, attr, attr_randomization_params,
                                        self.last_step, smpl)
                                else:
                                    set_random_properties = False
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get('setup_only', False)
                            if (setup_only and not self.sim_initialized) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], prop, attr)
                                apply_random_samples(
                                    prop, self.original_props[prop_name], attr,
                                    attr_randomization_params, self.last_step, smpl)
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False



if __name__ == "__main__":
    num_envs = 100
    physics_engine =  'physx'
    rl_device = 'cuda:0'
    sim_device = 'cuda:0'
    graphics_device_id = 0
    headless = False

    sim_params = dict(
        dt = 0.0166, # 1/60,
        substeps = 1,
        up_axis = "z",
        use_gpu_pipeline = True,
        gravity = [0.0, 0.0, -9.81],
        physx ={
            'num_threads': 4,
            'solver_type': 1,
            'use_gpu': True, # set to False to run on CPU
            'num_position_iterations': 12,
            'num_velocity_iterations': 1,
            'contact_offset': 0.005,
            'rest_offset': 0.0,
            'bounce_threshold_velocity': 0.2,
            'max_depenetration_velocity': 1000.0,
            'default_buffer_size_multiplier': 5.0,
            'max_gpu_contact_pairs': 1048576, # 1024*1024
            'num_subscenes': 1,
            'contact_collection': 0 # 0: CC_NEVER (don't collect contact info), 1: CC_LAST_SUBSTEP (collect only contacts on last substep), 2: CC_ALL_SUBSTEPS (broken - do not use!)
        }

    )

    cfg = {
        'physics_engine': 'physx',
        'sim': dict(
            dt = 0.0166, # 1/60,
            substeps = 1,
            up_axis = "z",
            use_gpu_pipeline = True,
            gravity = [0.0, 0.0, -9.81],
            physx ={
                'num_threads': 4,
                'solver_type': 1,
                'use_gpu': True, # set to False to run on CPU
                'num_position_iterations': 12,
                'num_velocity_iterations': 1,
                'contact_offset': 0.005,
                'rest_offset': 0.0,
                'bounce_threshold_velocity': 0.2,
                'max_depenetration_velocity': 1000.0,
                'default_buffer_size_multiplier': 5.0,
                'max_gpu_contact_pairs': 1048576, # 1024*1024
                'num_subscenes': 1,
                'contact_collection': 0 # 0: CC_NEVER (don't collect contact info), 1: CC_LAST_SUBSTEP (collect only contacts on last substep), 2: CC_ALL_SUBSTEPS (broken - do not use!)
            },
        ),
        'env':{
            'numEnvs': 10,
            'numActions': 7,
        }
    }

    model = VecTask(cfg, rl_device, sim_device, graphics_device_id, headless)








# class IsaacModel(nn.Module):
#     metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

#     def __init__(
#             self,
#             urdf_path:str,
#             num_envs:int,
#             sim_params,
#             physics_engine: str,
#             device: str,
#             graphics_device: int,
#             control_space: str = 'vel',
#             env_spacing: float = 1.5,
#             headless: bool = False,

#     ):
#         super().__init__()
#         self.urdf_path = urdf_path
#         split_device = device.split(':')
#         self.device = device
#         self.device_type = split_device[0]
#         self.device_id = int(split_device[1]) if len(split_device) > 1 else 0
#         self.graphics_device = graphics_device
#         self.num_envs = num_envs
#         self.control_space = control_space
#         self.env_spacing = env_spacing
#         self.headless = headless

#         self.clip_obs = np.Inf
#         self.clip_actions = np.Inf
#         self.action_scale = 1.0
#         self.num_states = 21


#         if physics_engine == "physx":
#             self.physics_engine = gymapi.SIM_PHYSX
#         elif physics_engine == "flex":
#             self.physics_engine = gymapi.SIM_FLEX
#         else:
#             msg = f"Invalid physics engine backend: {physics_engine}"
#             raise ValueError(msg)

#         self.sim_params = self._parse_sim_params(physics_engine, sim_params)
#         self.sim_params.up_axis = gymapi.UP_AXIS_Z
#         self.sim_params.gravity.x = 0
#         self.sim_params.gravity.y = 0
#         self.sim_params.gravity.z = -9.81

#         # optimization flags for pytorch JIT
#         torch._C._jit_set_profiling_mode(False)
#         torch._C._jit_set_profiling_executor(False)

#         self.gym = gymapi.acquire_gym()
#         # self.sim = self.gym.create_sim(
#         #     compute_device=self.device_id, 
#         #     graphics_device=self.graphics_device, 
#         #     type=self.physics_engine, 
#         #     params=self.sim_params)
#         self.sim_initialized = False
#         self.sim = self.create_sim(self.device_id, self.graphics_device, self.physics_engine, self.sim_params)
#         self._create_ground_plane()
#         self.create_envs(self.num_envs, self.env_spacing)
#         self.gym.prepare_sim(self.sim)
#         self.sim_initialized = True

#         self.set_viewer()
#         self.allocate_buffers()
#         # self.init_tensors()

#     def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
#         """Create an Isaac Gym sim object.

#         Args:
#             compute_device: ID of compute device to use.
#             graphics_device: ID of graphics device to use.
#             physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
#             sim_params: sim params to use.
#         Returns:
#             the Isaac Gym sim object.
#         """
#         sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
#         if sim is None:
#             print("*** Failed to create sim")
#             quit()

#         return sim

#     def _create_ground_plane(self):
#         plane_params = gymapi.PlaneParams()
#         plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
#         self.gym.add_ground(self.sim, plane_params)

#     def create_envs(self, num_envs, env_spacing):
#         # self._create_ground_plane()
#         # self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
#         lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
#         upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
#         num_per_row = int(np.sqrt(env_spacing))

#         asset_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../content/assets"))
#         franka_asset_file = "urdf/franka_description/franka_panda_no_gripper.urdf"


#         #load franka
#         asset_options = gymapi.AssetOptions()
#         asset_options.flip_visual_attachments = True
#         asset_options.fix_base_link = True
#         asset_options.collapse_fixed_joints = True
#         asset_options.disable_gravity = True
#         asset_options.thickness = 0.001
#         asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
#         asset_options.use_mesh_materials = True
#         franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)   
        
#         # franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.compute_device)
#         # franka_dof_damping = to_torch([40, 40, 40, 40, 40, 40, 40, 1.0e2, 1.0e2], dtype=torch.float, device=self.compute_device)

#         franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
#         franka_dof_damping = to_torch([40, 40, 40, 40, 40, 40, 40], dtype=torch.float, device=self.device)


#         self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
#         self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)


#         franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
#         self.franka_dof_lower_limits = []
#         self.franka_dof_upper_limits = []
#         for i in range(self.num_franka_dofs):
#             franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
#             if self.physics_engine == gymapi.SIM_PHYSX:
#                 franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
#                 franka_dof_props['damping'][i] = franka_dof_damping[i]
#             else:
#                 franka_dof_props['stiffness'][i] = 7000.0
#                 franka_dof_props['damping'][i] = 50.0

#             self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
#             self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

#         self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
#         self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
#         self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

#         self.franka_pose_world = gymapi.Transform()
#         self.franka_pose_world.p = gymapi.Vec3(0.0, 0.0, 0.0)
#         self.franka_pose_world.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


#         self.frankas = []
#         self.envs = []

#         for i in range(num_envs):
#             # create env instance
#             env_ptr = self.gym.create_env(
#                 self.sim, lower, upper, num_per_row
#             )

#             # if self.aggregate_mode >= 3:
#             # self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
#             # table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose_world, "table", i, 1, 0)
#             # self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

#             franka_actor = self.gym.create_actor(env_ptr, franka_asset, self.franka_pose_world, "franka", i, 1, 0)
#             self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
#             self.envs.append(env_ptr)
#             # self.tables.append(table_actor)
#             self.frankas.append(franka_actor)
        
#         self.ee_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "ee_link")

#     def forward(self, actions):
#         return self.step(actions)

#     def pre_physics_step(self, actions: torch.Tensor):
#         """Apply the actions to the environment (eg by setting torques, position targets).

#         Args:
#             actions: the actions to apply
#         """
#         self.actions = actions.clone().to(self.device)
#         if self.control_space == "pos":
#             targets = self.actions
#         elif self.control_space == "vel":
#             targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
#         elif self.control_space == "vel_2":
#             targets = self.franka_dof_pos + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
#         elif self.control_space == "acc":
#             raise NotImplementedError

#         # targets = actions #self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
#         self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
#             targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
#         # env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)        
#         self.gym.set_dof_position_target_tensor(self.sim,
#                                                 gymtorch.unwrap_tensor(self.franka_dof_targets))

        
#     def post_physics_step(self):
#         """Compute reward and observations, reset any environments that require it."""
#         self.progress_buf += 1
#         env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
#         if len(env_ids) > 0:
#             self.reset_idx(env_ids)
#         # if self.viewer:
#         #     self.gym.clear_lines(self.viewer)
#         #     for i in range(self.num_envs):
#         #         #plot target axes
#         #         axes_geom = gymutil.AxesGeometry(0.1)
#         #         # Create a wireframe sphere
#         #         sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * np.pi, 0, 0)
#         #         sphere_pose = gymapi.Transform(r=sphere_rot)
#         #         sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(0, 1, 0))
#         #         target_pos = self.target_poses[i, 0:3]
#         #         target_rot = self.target_poses[i, 3:7]
#         #         target_pos = gymapi.Vec3(x=target_pos[0], y=target_pos[1], z=target_pos[2]) 
#         #         target_rot = gymapi.Quat(x=target_rot[1],y=target_rot[2], z=target_rot[3], w=target_rot[0])
#         #         target_pose_franka = gymapi.Transform(p=target_pos, r=target_rot)
#         #         target_pose_world = target_pose_franka * self.franka_pose_world
#         #         gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], target_pose_world)
#         #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], target_pose_world)
#                 # #plot ee axes
#                 # ee_pos = self.rigid_body_states[i, self.ee_handle][0:3]
#                 # ee_rot = self.rigid_body_states[i, self.ee_handle][3:7]
#                 # ee_pos = gymapi.Vec3(x=ee_pos[0], y=ee_pos[1], z=ee_pos[2])
#                 # ee_rot = gymapi.Quat(x=ee_rot[0],y=ee_rot[1], z=ee_rot[2], w=ee_rot[3])
#                 # ee_pose_world = gymapi.Transform(p=ee_pos, r=ee_rot)
#                 # axes_geom = gymutil.AxesGeometry(0.1)
#                 # sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * np.pi, 0, 0)
#                 # sphere_pose = gymapi.Transform(r=sphere_rot)
#                 # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 1, 0))
#                 # gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], ee_pose_world)
#                 # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], ee_pose_world)
#         # print('draw time', time.time()-st)



#     def step(self, actions):

#         action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
#         # apply actions
#         self.pre_physics_step(action_tensor)

#         # step physics and render each frame
#         for i in range(self.control_freq_inv):
#             if self.force_render:
#                 self.render()
#             self.gym.simulate(self.sim)

#         # to fix!
#         if self.device == 'cpu':
#             self.gym.fetch_results(self.sim, True)

#         # compute observations, rewards, resets, ...
#         self.post_physics_step()

#         # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
#         self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

#         # # randomize observations
#         # if self.dr_randomizations.get('observations', None):
#         #     self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

#         self.extras["time_outs"] = self.timeout_buf.to(self.device)

#         # self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

#         # asymmetric actor-critic
#         # if self.num_states > 0:
#         #     self.obs_dict["states"] = self.get_state()

#         # return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
#         return self.states_buf.to(self.device), \
#                self.reset_buf.to(self.device)


#     def allocate_buffers(self):
#         """Allocate the observation, states, etc. buffers.

#         These are what is used to set observations and states in the environment classes which
#         inherit from this one, and are read in `step` and other related functions.

#         """

#         # allocate buffers
#         # self.obs_buf = torch.zeros(
#         #     (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
#         self.states_buf = torch.zeros(
#             (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
#         # self.rew_buf = torch.zeros(
#         #     self.num_envs, device=self.device, dtype=torch.float)
#         self.reset_buf = torch.ones(
#             self.num_envs, device=self.device, dtype=torch.long)
#         self.timeout_buf = torch.zeros(
#              self.num_envs, device=self.device, dtype=torch.long)
#         self.progress_buf = torch.zeros(
#             self.num_envs, device=self.device, dtype=torch.long)
#         self.randomize_buf = torch.zeros(
#             self.num_envs, device=self.device, dtype=torch.long)
#         self.extras = {}

#     def init_tensors(self):
#         actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
#         dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
#         rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
#         jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "franka")


#         self.gym.refresh_actor_root_state_tensor(self.sim)
#         self.gym.refresh_dof_state_tensor(self.sim)
#         self.gym.refresh_rigid_body_state_tensor(self.sim)
#         self.gym.refresh_jacobian_tensors(self.sim)

#         # create some wrapper tensors for different slices
#         # self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469], device=self.device)
#         self.franka_default_dof_pos = to_torch([0.0, -0.7853, 0.0, -2.3561, 0.0, 1.5707, 0.7853], device=self.device)
#         self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
#         self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
#         self.franka_dof_pos = self.franka_dof_state[..., 0]
#         self.franka_dof_vel = self.franka_dof_state[..., 1]
#         self.franka_dof_acc = torch.zeros_like(self.franka_dof_vel)
#         self.tstep = torch.ones(self.num_envs, 1, device=self.device)
#         self.franka_jacobian = gymtorch.wrap_tensor(jacobian_tensor)

#     def set_viewer(self):
#         """Create the viewer."""

#         self.enable_viewer_sync = True
#         self.viewer = None

#         # if running with a viewer, set up keyboard shortcuts and camera
#         if self.headless == False:
#             # subscribe to keyboard shortcuts
#             self.viewer = self.gym.create_viewer(
#                 self.sim, gymapi.CameraProperties())
#             self.gym.subscribe_viewer_keyboard_event(
#                 self.viewer, gymapi.KEY_ESCAPE, "QUIT")
#             self.gym.subscribe_viewer_keyboard_event(
#                 self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

#             # set the camera position based on up axis
#             sim_params = self.gym.get_sim_params(self.sim)
#             if sim_params.up_axis == gymapi.UP_AXIS_Z:
#                 cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
#                 cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
#             else:
#                 cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
#                 cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

#             self.gym.viewer_camera_look_at(
#                 self.viewer, None, cam_pos, cam_target)

#     def _parse_sim_params(self, physics_engine, param_dict):
#         sim_params = gymapi.SimParams()

#         # check correct up-axis
#         if param_dict["up_axis"] not in ["z", "y"]:
#             msg = f"Invalid physics up-axis: {param_dict['up_axis']}"
#             print(msg)
#             raise ValueError(msg)

#         # assign general sim parameters
#         sim_params.dt = param_dict["dt"]
#         sim_params.num_client_threads = param_dict.get("num_client_threads", 0)
#         sim_params.use_gpu_pipeline = param_dict["use_gpu_pipeline"]
#         sim_params.substeps = param_dict.get("substeps", 2)

#         # assign up-axis
#         if param_dict["up_axis"] == "z":
#             sim_params.up_axis = gymapi.UP_AXIS_Z
#         else:
#             sim_params.up_axis = gymapi.UP_AXIS_Y

#         # assign gravity
#         sim_params.gravity = gymapi.Vec3(*param_dict["gravity"])

#         # configure physics parameters
#         if physics_engine == "physx":
#             # set the parameters
#             if "physx" in param_dict:
#                 for opt in param_dict["physx"].keys():
#                     if opt == "contact_collection":
#                         setattr(sim_params.physx, opt, gymapi.ContactCollection(param_dict["physx"][opt]))
#                     else:
#                         setattr(sim_params.physx, opt, param_dict["physx"][opt])
#         else:
#             # set the parameters
#             if "flex" in param_dict:
#                 for opt in param_dict["flex"].keys():
#                     setattr(sim_params.flex, opt, param_dict["flex"][opt])

#         # return the configured params
#         return sim_params


# if __name__ == "__main__":
#     urdf_path = "urdf/franka_description/franka_panda_no_gripper.urdf"
#     num_envs = 100
#     physics_engine =  'physx'
#     sim_device = 'cuda:0'
#     graphics_device_id = 1

#     sim_params = dict(
#         dt = 0.0166, # 1/60,
#         substeps = 1,
#         up_axis = "z",
#         use_gpu_pipeline = True,
#         gravity = [0.0, 0.0, -9.81],
#         physx ={
#             'num_threads': 4,
#             'solver_type': 1,
#             'use_gpu': True, # set to False to run on CPU
#             'num_position_iterations': 12,
#             'num_velocity_iterations': 1,
#             'contact_offset': 0.005,
#             'rest_offset': 0.0,
#             'bounce_threshold_velocity': 0.2,
#             'max_depenetration_velocity': 1000.0,
#             'default_buffer_size_multiplier': 5.0,
#             'max_gpu_contact_pairs': 1048576, # 1024*1024
#             'num_subscenes': 1,
#             'contact_collection': 0 # 0: CC_NEVER (don't collect contact info), 1: CC_LAST_SUBSTEP (collect only contacts on last substep), 2: CC_ALL_SUBSTEPS (broken - do not use!)
#         }

#     )

#     model = IsaacModel(
#         urdf_path,
#         num_envs,
#         sim_params,
#         physics_engine,
#         sim_device, graphics_device_id,
#         control_space='vel',
#         headless=False
#         )

#     input('....')