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

from storm_kit.differentiable_robot_model.coordinate_transform import CoordinateTransform, quaternion_to_matrix, matrix_to_quaternion, rpy_angles_to_matrix
from storm_kit.envs.env_utils import tensor_clamp

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class PointRobotEnv(): 
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
        # self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        # self.control_space = self.cfg["env"]["controlSpace"]
        self.num_environments = self.cfg["env"]["num_envs"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.up_axis = "z"
        self.up_axis_idx = 2
        self.dt = self.cfg["sim"]["dt"]
        self.world_params = self.cfg["world"]
        self.world_model = self.world_params["world_model"]
        self.num_objects = self.cfg["env"]["num_objects"]
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
        # self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.reset()
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
                cam_pos = gymapi.Vec3(0.0, 1.0, 1.0)
                cam_target = gymapi.Vec3(0.0, -1.0, 0.0)
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

        self.target_buf = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.start_buf = torch.zeros(
            self.num_envs, 3, device=self.device)
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

        robot_asset, robot_dof_props = self.load_robot_asset()
        table_asset, table_dims, table_color = self.load_table_asset()

        table_pose_world = gymapi.Transform()
        table_pose_world.p = gymapi.Vec3(0, 0, 0 + table_dims.z)
        table_pose_world.r = gymapi.Quat(0., 0., 0., 1.)
        self.robot_start_pose_table = gymapi.Transform()
        # self.robot_start_pose_table.p = gymapi.Vec3(-table_dims.x/2.0 + 0.3, 0.0, table_dims.z/2.0 + 0.03 + 0.01)
        self.robot_start_pose_table.p = gymapi.Vec3(0.0, 0.0, table_dims.z/2.0 + 0.03 + 0.01)

        self.robot_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.robot_pose_world =  table_pose_world * self.robot_start_pose_table #convert from franka to world frame

        self.num_object_bodies = 0
        self.num_object_shapes = 0

        if self.num_objects > 0:
            object_assets = []
            for _ in range(self.num_objects):
                object_asset, object_color = self.load_object_asset(disable_gravity=False)
                self.num_object_bodies += self.gym.get_asset_rigid_body_count(object_asset)
                self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

                object_assets.append(object_asset)
                ball_start_pose_table = gymapi.Transform()
                # ball_start_pose_table.p = gymapi.Vec3(-table_dims.x/2.0 + 0.3 + 0.02 + 0.01 + 0.01, 0.0, table_dims.z/2.0 + 0.02) #0.3
                ball_start_pose_table.p = gymapi.Vec3(0.02 + 0.01 + 0.01, 0.0, table_dims.z/2.0 + 0.02) #0.3
                ball_start_pose_table.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            self.ball_start_pose_world =  table_pose_world * ball_start_pose_table #convert from franka to world frame

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
        self.world_pose_robot = temp.inverse() #convert from world frame to robot

        # compute aggregate size
        max_agg_bodies = self.num_robot_bodies + 1 + self.num_object_bodies #+ self.num_props * num_prop_bodies
        max_agg_shapes = self.num_robot_shapes + 1 + self.num_object_shapes #+ num_target_shapes #+ self.num_props * num_prop_shapes

        # self.tables = []
        self.robots = []
        self.envs = []
        self.objects = []

        for i in range(self.num_envs):
            # create env instance
            env_objects = []
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, self.robot_pose_world, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)            
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose_world, "table", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.num_objects > 0:
                for i, object_asset in enumerate(object_assets):
                    object_actor = self.gym.create_actor(env_ptr, object_asset, self.ball_start_pose_world, "ball_{}".format(i), i, 0, 0)
                    self.gym.set_rigid_body_color(env_ptr, object_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, object_color)
                    body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_actor)
                    for b in range(len(body_props)):
                        body_props[b].flags = gymapi.RIGID_BODY_NONE
                    self.gym.set_actor_rigid_body_properties(env_ptr, object_actor, body_props)
                    env_objects.append(object_actor)

            # self.ball_actor = self.gym.create_actor(env_ptr, ball_asset, self.ball_start_pose_world, "ball", i, 0, 0)
            # self.gym.set_rigid_body_color(env_ptr, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, ball_color)
            # body_props = self.gym.get_actor_rigid_body_properties(env_ptr, self.ball_actor)
            # for b in range(len(body_props)):
            #     body_props[b].flags = gymapi.RIGID_BODY_NONE
            # self.gym.set_actor_rigid_body_properties(env_ptr, self.ball_actor, body_props)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.objects.append(env_objects)

        
        self.init_data()
    
    def init_data(self):
        
        # self.ee_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "ee_link")
        self._refresh()
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
        
        
        self.robot_default_dof_pos = to_torch([0.0]* self.num_robot_dofs, device=self.device)
        self.robot_dof_state = self.dof_state[:, :self.num_robot_dofs]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]
        self.robot_dof_acc = torch.zeros_like(self.robot_dof_vel)
        self.tstep = torch.ones(self.num_envs, 1, device=self.device)

        self.robot_state = self.rigid_body_states[:, 2]
        self.init_robot_state = torch.zeros(self.num_envs, 13, device=self.device)
        self.init_robot_state[:,0] = self.robot_pose_world.p.x
        self.init_robot_state[:,1] = self.robot_pose_world.p.y
        self.init_robot_state[:,2] = self.robot_pose_world.p.z
        self.init_robot_state[:,3] = self.robot_pose_world.r.x
        self.init_robot_state[:,4] = self.robot_pose_world.r.y
        self.init_robot_state[:,5] = self.robot_pose_world.r.z
        self.init_robot_state[:,6] = self.robot_pose_world.r.w


        #Note: this needs to change to support more than one object!!!
        if self.num_objects > 0:
            self.object_state = self.root_state[:,-1]
            self.init_object_state = torch.zeros(self.num_envs, 13, device=self.device)
            self.init_object_state[:,0] = self.ball_start_pose_world.p.x
            self.init_object_state[:,1] = self.ball_start_pose_world.p.y
            self.init_object_state[:,2] = self.ball_start_pose_world.p.z
            self.init_object_state[:,3] = self.ball_start_pose_world.r.x
            self.init_object_state[:,4] = self.ball_start_pose_world.r.y
            self.init_object_state[:,5] = self.ball_start_pose_world.r.z
            self.init_object_state[:,6] = self.ball_start_pose_world.r.w

        self.num_bodies = self.rigid_body_states.shape[1]
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, 
                                            device=self.device).view(self.num_envs, -1)
        

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
        
    def load_robot_asset(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../content/assets")
        robot_asset_file = "urdf/point_robot.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            robot_asset_file = self.cfg["env"]["asset"].get("assetFileNameRobot", robot_asset_file)
        
        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        # set franka dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        robot_dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
        robot_dof_props["stiffness"].fill(0.0)
        robot_dof_props["damping"].fill(1.0)

        #Save robot dof limits
        self.robot_q_pos_lower_lims = []
        self.robot_q_pos_upper_lims = []
        self.robot_q_vel_lims = []
        self.robot_effort_lims = []
        for i in range(self.num_robot_dofs):
            self.robot_q_pos_lower_lims.append(robot_dof_props['lower'][i])
            self.robot_q_pos_upper_lims.append(robot_dof_props['upper'][i])
            self.robot_q_vel_lims.append(robot_dof_props['velocity'][i])
            self.robot_effort_lims.append(robot_dof_props['effort'][i])
        self.robot_q_pos_lower_lims = torch.tensor(self.robot_q_pos_lower_lims, device=self.device).unsqueeze(0).repeat(self.num_envs,1)
        self.robot_q_pos_upper_lims = torch.tensor(self.robot_q_pos_upper_lims, device=self.device).unsqueeze(0).repeat(self.num_envs,1)
        self.robot_q_vel_lims = torch.tensor(self.robot_q_vel_lims, device=self.device).unsqueeze(0).repeat(self.num_envs,1)
        self.robot_effort_lims = torch.tensor(self.robot_effort_lims, device=self.device).unsqueeze(0).repeat(self.num_envs,1)

        return robot_asset, robot_dof_props


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


    def pre_physics_step(self, actions: torch.Tensor):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        pos_des = actions[:, 0:self.num_dofs].clone().to(self.device)
        vel_des = actions[:, self.num_dofs:2*self.num_dofs].clone().to(self.device)
        acc_des = actions[:, 2*self.num_dofs:3*self.num_dofs].clone().to(self.device)
        
        if pos_des.ndim == 3:
            pos_des = pos_des[:, 0]
            vel_des = vel_des[:, 0]
            acc_des = acc_des[:, 0]
        
        curr_robot_pos = self.robot_state[:, 0:2]
        curr_robot_vel = self.robot_state[:, 7:9]
        feedforward_torques = torch.einsum('ijk,ik->ij', self.robot_mass, acc_des)
        feedback_torques =  10.0 * (pos_des - curr_robot_pos) + 1.0 * (vel_des - curr_robot_vel)      

        torques = feedforward_torques + feedback_torques
        torques = tensor_clamp(torques, min=-1.*self.robot_effort_lims, max=self.robot_effort_lims)
        
        
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))


    # def pre_physics_step(self, action_dict: Dict[str, torch.tensor]):
    #     # implement pre-physics simulation code here
    #     #    - e.g. apply actions
    #     if 'raw_action' in action_dict:
    #         actions = action_dict['raw_action'].clone().to(self.device)
    #         if actions.ndim == 3:
    #             actions = actions[:, 0]
    #     else:
    #         pos_des = action_dict['q_pos'].clone().to(self.device)
    #         vel_des = action_dict['q_vel'].clone().to(self.device)
    #         acc_des = action_dict['q_acc'].clone().to(self.device)
    #         if pos_des.ndim == 3:
    #             pos_des = pos_des[:, 0]
    #             vel_des = vel_des[:, 0]
    #             acc_des = acc_des[:, 0]
    #         curr_robot_pos = self.robot_state[:, 0:2]
    #         curr_robot_vel = self.robot_state[:, 7:9]
    #         actions =  -1.0 * (curr_robot_pos - pos_des) - 0.01 * (curr_robot_vel- vel_des)      
    #     actions = tensor_clamp(actions, min=-1.*self.robot_effort_lims, max=self.robot_effort_lims)
    #     self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(actions))

    def step(self, actions: Dict[str, torch.tensor]): # -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:

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


        return state_dict, self.reset_buf.to(self.rl_device)


    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1                
        # env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
        #     print('inside resetting envs')
        #     self.reset_idx(env_ids)
        state_dict = self.get_state_dict()
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        return state_dict

    def get_state_dict(self):
        self._refresh()
        # self.robot_q_pos_buff[:] = self.robot_dof_pos
        # self.robot_q_vel_buff[:] = self.robot_dof_vel
        # self.robot_q_acc_buff[:] = self.robot_dof_acc
        tstep = self.gym.get_sim_time(self.sim)
        tstep *= self.tstep
        robot_pos = self.robot_state[:, 0:3]
        robot_vel = self.robot_state[:, 7:10]
        robot_acc = self.robot_dof_acc

        state_dict = {
            'q_pos': robot_pos[:,0:2].to(self.rl_device),
            'q_vel': robot_vel[:,0:2].to(self.rl_device),
            'q_acc': robot_acc[:,0:2].to(self.rl_device),
            'tstep': tstep
        }
        if self.num_objects > 0:
            #Note: This won't work for more than one object
            object_pos = self.object_state[:,0:3]
            object_rot = self.object_state[:,3:7]
            object_vel = self.object_state[:,7:10]
            object_ang_vel = self.object_state[:,10:13]

            state_dict['object_pos'] = object_pos[:,0:2].to(self.rl_device)
            state_dict['object_rot'] = object_rot[:,0:2].to(self.rl_device)
            state_dict['object_vel'] = object_vel[:,0:2].to(self.rl_device)
            state_dict['object_ang_vel'] = object_ang_vel[:,0:2].to(self.rl_device)


        return state_dict

    def reset(self, reset_data=None):
        _ = self.reset_idx(torch.arange(self.num_envs, device=self.device), reset_data=reset_data)
        state_dict = self.get_state_dict()
        return state_dict 


    def reset_idx(self, env_ids, reset_data=None):

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # reset robot
        pos = self.robot_default_dof_pos.unsqueeze(0)
        self.robot_dof_pos[env_ids, :] = pos
        self.robot_dof_vel[env_ids, :] = torch.zeros_like(self.robot_dof_vel[env_ids])

        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()
        multi_env_ids_object_int32 = self.global_indices[env_ids, -1].flatten()

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))         

        if self.num_objects > 0:
            self.object_state[env_ids] = self.init_object_state[env_ids].clone()
                

        if reset_data is not None:
            if 'goal_dict' in reset_data:
                self.update_goal(reset_data['goal_dict'])
            
            if 'start_dict' in reset_data:
                self.update_start(reset_data['start_dict'])
                if self.num_objects > 0:
                    #reset object
                    self.object_state[env_ids, 0:2] = self.start_buf[env_ids][env_ids, 0:2].clone()
                    self.object_state[env_ids, 7:9] = torch.zeros(self.num_envs, 2, device=self.rl_device)
                    pos = self.robot_default_dof_pos.unsqueeze(0)
                    self.robot_dof_pos[env_ids, :] = pos
                    self.robot_dof_vel[env_ids, :] = torch.zeros_like(self.robot_dof_vel[env_ids])
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim, gymtorch.unwrap_tensor(self.root_state),
                        gymtorch.unwrap_tensor(multi_env_ids_object_int32), len(multi_env_ids_object_int32))

                # else:
                #     self.robot_state[env_ids, 0:2] = self.start_buf[env_ids, 0:2].clone()
                #     self.robot_dof_pos[env_ids, :] =  self.robot_default_dof_pos.unsqueeze(0)
                #     self.robot_dof_vel[env_ids, :] = torch.zeros_like(self.robot_dof_vel[env_ids])


        

        #update buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
    

        state_dict = self.get_state_dict()
        return state_dict 

    def update_goal(self, goal_dict):
        if 'object_goal' in goal_dict:
            self.target_buf = goal_dict['object_goal']
        else:
            self.target_buf = goal_dict['robot_goal']

    def update_start(self, start_dict):
        if 'object_start_pos' in start_dict:
            self.start_buf = start_dict['object_start_pos']
        else:
            self.start_buf = start_dict['robot_start_pos']
    

        
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
                # Plot sphere at target
                sphere_rot = gymapi.Quat.from_euler_zyx(0.0, 0, 0)
                sphere_pose = gymapi.Transform(r=sphere_rot)
                sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(0, 1, 0))
                target_pos = self.target_buf[i, 0:2]
                if self.num_objects > 0:
                    z = self.init_object_state[i,2]
                else:
                    z = self.robot_pose_world.p.z
                target_pos = gymapi.Vec3(x=target_pos[0], y=target_pos[1], z=z) 
                target_pose = gymapi.Transform(p=target_pos)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], target_pose)


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



    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    # @property
    # def num_acts(self) -> int:
    #     """Get the number of actions in the environment."""
    #     return self.num_actions

    # @property
    # def num_robot_dofs(self):
    #     return self.num_robot_dofs