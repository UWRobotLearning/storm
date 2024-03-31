#!/usr/bin/env python
import copy, os
import torch
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
import time

from storm_kit.learning.policies import MPCPolicy, JointControlWrapper
from storm_kit.tasks import ArmReacher
from storm_kit.mpc.model import URDFKinematicModel
from storm_kit.mpc.utils.state_filter import JointStateFilter
from storm_kit.learning.learning_utils import dict_to_device


class PandaRealRobotEnv():
    def __init__(
            self, cfg, device=torch.device('cpu'), 
            headless:bool=False, safe_mode:bool = True, launch_name:str="robot_world_publisher",
            auto_reset_on_episode_end:bool = True):
        self.cfg = cfg
        self.max_episode_length = cfg['env']['episodeLength']
        self.n_dofs = cfg.joint_control.n_dofs #for compatibility with new configs
        self.device = device
        self.num_envs = cfg.env.get('num_envs', 1)
        self.robot_default_dof_pos = self.cfg["env"]["robot_default_dof_pos"]
        self.headless = headless
        self.safe_mode = safe_mode
        self.auto_reset_on_episode_end = auto_reset_on_episode_end
        rospy.init_node("panda_real_robot_env", anonymous=True, disable_signals=False)    

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.joint_names = rospy.get_param('~robot_joint_names', ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        self.control_dt = 0.02 #self.cfg.joint_control.control_dt
        self.robot_default_dof_pos = torch.tensor(self.robot_default_dof_pos, device=self.device).unsqueeze(0)

        self.mpc_command = JointState()
        self.mpc_command.name = self.joint_names
        self.fixed_frame = 'base_link'

        self.command_header = None
        self.gripper_state = {
            'q_pos': torch.zeros(1,2),
            'q_vel': torch.zeros(1,2),
            'q_acc': torch.zeros(1,2)}
        self.robot_state = {
            'q_pos': torch.zeros(1,7),
            'q_vel': torch.zeros(1,7),
            'q_acc': torch.zeros(1,7)}
        # self.robot_state['q_acc'] = torch.zeros_like(self.robot_state['q_vel'])

        self.robot_state_tensor = None

        #ROS Initialization
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)

        self.control_freq = float(1.0/self.control_dt)
        self.rate = rospy.Rate(self.control_freq)

        self.state_sub_on = False
        self.tstep = 0
        self.start_t = None
        self.first_iter = True

        self.allocate_buffers()
        self.state_filter = JointStateFilter(
            filter_coeff=self.cfg.joint_control.state_filter_coeff, 
            device=self.device,
            n_dofs=self.n_dofs,
            dt=self.cfg.joint_control.control_dt)

        self.init_reset_policy()
        if not self.headless:
            import roslaunch
            import tf2_ros
            import rospkg
            from geometry_msgs.msg import TransformStamped
            rospack = rospkg.RosPack()
            self.pkg_path = rospack.get_path('storm_ros')

            self.br = tf2_ros.TransformBroadcaster()
            self.goal_transform = TransformStamped()

            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            launch = roslaunch.parent.ROSLaunchParent(
                uuid, [os.path.join(self.pkg_path, "launch/{}.launch".format(launch_name))])
            launch.start()


    def robot_state_callback(self, msg):
        self.state_sub_on = True
        self.command_header = msg.header
        #save gripper state
        # self.gripper_state.header = msg.header
        # self.gripper_state.position = msg.position[0:2]
        # self.gripper_state.velocity = msg.velocity[0:2]
        # self.gripper_state.effort = msg.effort[0:2]

        # self.gripper_state['position'] = np.array(msg.position[0:2])
        # self.gripper_state['velocity'] = np.array(msg.velocity[0:2])

        # #save robot state
        # self.robot_state.header = msg.header
        # self.robot_state.position = msg.position[2:]
        # self.robot_state.velocity = msg.velocity[2:]
        # self.robot_state.effort = msg.effort[2:]
        self.robot_state['q_pos'] = torch.tensor(msg.position).unsqueeze(0)
        self.robot_state['q_vel'] = torch.tensor(msg.velocity).unsqueeze(0)

        # self.robot_state_tensor = torch.cat((
        #     self.robot_state['q_pos'],
        #     self.robot_state['q_vel'],
        #     self.robot_state['q_acc']
        # )).unsqueeze(0)


    def allocate_buffers(self):
        # allocate buffers
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.target_buf = torch.zeros(
            (self.num_envs, 7), device=self.device
        )

    def _create_envs(self):
        pass

    def init_data(self):
        pass

    def pre_physics_steps(self, actions:torch.Tensor):
        #XXX Remove deepcopies
        command_dict = copy.deepcopy(
            self.state_filter.predict_internal_state(actions))

        return command_dict

    def step(self, actions:torch.Tensor):
        
        if not self.headless:
            self.render()

        #only publish command if state has been received
        if self.state_sub_on:
            #convert actions to desired command
            command_dict = self.pre_physics_steps(actions)
            #publish mpc 
            self.mpc_command.header = self.command_header
            self.mpc_command.header.stamp = rospy.Time.now()
            self.mpc_command.position = command_dict['q_pos'][0].cpu().numpy()
            self.mpc_command.velocity = command_dict['q_vel'][0].cpu().numpy()
            self.mpc_command.effort =  command_dict['q_acc'][0].cpu().numpy()

            # self.mpc_command.position = actions[0][0:7].cpu().numpy()
            # self.mpc_command.velocity = actions[0][7:14].cpu().numpy()
            # self.mpc_command.effort =  actions[0][14:21].cpu().numpy()
            self.command_pub.publish(self.mpc_command)
            
            #update tstep
            if self.tstep == 0:
                rospy.loginfo('[PandaRobotEnv]: Env Setup')
                self.start_t = rospy.get_time()
            self.tstep = rospy.get_time() - self.start_t

        else:
            if (not self.state_sub_on) and (self.first_iter):
                rospy.loginfo('[PandaRobotEnv]: Waiting for robot state.')
        
        self.first_iter = False
        self.rate.sleep()
        state_dict = self.post_physics_step() 

        return state_dict, self.reset_buf
    

    def post_physics_step(self):
        self.progress_buf += 1                
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0 and self.auto_reset_on_episode_end:
            self.reset()
        state_dict = self.get_state_dict()
        state_dict = self.state_filter.filter_joint_state(copy.deepcopy(state_dict)) #XXX remove deepcopy?

        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length, torch.ones_like(self.reset_buf), self.reset_buf)

        return state_dict

    def get_state_dict(self):
        #XX remove deepcopy
        #XX check dict to device
        state_dict = copy.deepcopy(self.robot_state)
        state_dict['tstep'] = torch.as_tensor([self.tstep]).unsqueeze(0)
        state_dict = dict_to_device(state_dict, self.device)
        return state_dict

    def reset(self, reset_data=None):
        if reset_data is not None:
            if 'goal_dict' in reset_data:
                self.update_goal(reset_data['goal_dict'])
        if self.safe_mode:
            input('[PandaRobotEnv]: Press enter to begin reset')
        
        print('[PandaRobotEnv]: Resetting to default joint config')
        reset_data = {}
        reset_data['goal_dict'] = dict(joint_goal = self.robot_default_dof_pos)
        self.reset_policy.reset(reset_data)
        max_steps = 500
        curr_q_pos = self.robot_state['q_pos']
        q_pos_goal = self.robot_default_dof_pos.cpu()
        curr_error = torch.norm(curr_q_pos - q_pos_goal, p=2)
        curr_num_steps = 0
        tstep, start_t = 0, 0

        while True:
            try:
                policy_input = {
                    'states': self.get_state_dict()}
                action, policy_info = self.reset_policy.get_action(policy_input, deterministic=True)
                command_tensor = policy_info['command']

                self.mpc_command.header = self.command_header
                self.mpc_command.header.stamp = rospy.Time.now()
                self.mpc_command.position = command_tensor[0][0:7].cpu().numpy()
                self.mpc_command.velocity = command_tensor[0][7:14].cpu().numpy()
                self.mpc_command.effort =  command_tensor[0][14:21].cpu().numpy()
                self.command_pub.publish(self.mpc_command)
            
                #update tstep
                if tstep == 0:
                    start_t = rospy.get_time()
                tstep = rospy.get_time() - start_t

                curr_q_pos = self.robot_state['q_pos'].clone()
                curr_error = torch.norm(curr_q_pos - q_pos_goal, p=2).item()
                curr_num_steps += 1
                if (curr_error <= 0.005) or (curr_num_steps == max_steps -1):
                    print('[PandaRobotEnv]: Reset joint error = {}', curr_error)
                    break
                self.rate.sleep()
            except KeyboardInterrupt:
                self.close()

        self.progress_buf[:] = 0
        self.reset_buf[:] = 0
        if self.safe_mode: 
            input('[PandaRobotEnv]: Press enter to finish reset')
        state_dict = self.get_state_dict()
        self.state_filter.reset()
        state_dict = copy.deepcopy(self.state_filter.filter_joint_state(copy.deepcopy(state_dict)))
        return state_dict 

    def reset_filter(self, state_dict):
        self.state_filter.reset()
        state_dict = copy.deepcopy(self.state_filter.filter_joint_state(copy.deepcopy(state_dict)))
        return state_dict 
    
    def close(self):
        self.command_pub.unregister()
        self.state_sub.unregister()

    def init_reset_policy(self):

        reset_cfg = compose(config_name="config", overrides=["task=FrankaReacherRealRobot", "mpc=FrankaReacherRealRobotMPC"])

        mpc_config = reset_cfg.mpc
        # mpc_config.rollout.cost.goal_pose.weight = [0.0, 0.0]
        # mpc_config.rollout.cost.joint_l2.weight = 5.0
        # mpc_config.rollout.cost.ee_vel_twist.weight = 0.0
        # mpc_config.rollout.cost.zero_q_vel.weight = 0.1
        # mpc_config.rollout.cost.stop_cost.weight = 2.0
        mpc_config.task.cost.goal_pose.weight = [0.0, 0.0]
        mpc_config.task.cost.joint_l2.weight = 5.0
        mpc_config.task.cost.ee_vel_twist.weight = 0.0
        mpc_config.task.cost.zero_q_vel.weight = 0.1
        mpc_config.task.cost.stop_cost.weight = 2.0
        mpc_config.mppi.update_cov = False

        self.reset_policy = MPCPolicy(
            obs_dim=1, act_dim=1, 
            config=mpc_config, task_cls=ArmReacher, 
            dynamics_model_cls=URDFKinematicModel, device=self.device)
        self.reset_policy = JointControlWrapper(
            config=reset_cfg.task.joint_control, 
            policy=self.reset_policy, device=self.device)
    
    def render(self):
        #draw targets etc.
        self.draw_auxillary_visuals()

    def draw_auxillary_visuals(self):
        #publish goal transform
        if self.target_buf is not None:
            target_pos = self.target_buf[0, 0:3].cpu()
            target_quat = self.target_buf[0, 3:7].cpu()

            self.goal_transform.header.stamp = rospy.Time.now()
            self.goal_transform.header.frame_id = self.fixed_frame
            self.goal_transform.child_frame_id = 'ee_goal'
            self.goal_transform.transform.translation.x = target_pos[0].item()
            self.goal_transform.transform.translation.y = target_pos[1].item()
            self.goal_transform.transform.translation.z = target_pos[2].item()
            self.goal_transform.transform.rotation.x = target_quat[1].item()
            self.goal_transform.transform.rotation.y = target_quat[2].item()
            self.goal_transform.transform.rotation.z = target_quat[3].item()
            self.goal_transform.transform.rotation.w = target_quat[0].item()

            self.br.sendTransform(self.goal_transform)

    def update_goal(self, goal_dict):
        if 'object_goal' in goal_dict:
            self.target_buf = goal_dict['object_goal']
            self.target_type = 'object_goal'
        else:
            self.target_buf = goal_dict['ee_goal']
            self.target_type = 'robot_goal'


@hydra.main(config_name="config", config_path="../../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    print("rl device", cfg.rl_device)
    env = PandaRealRobotEnv(cfg.task, device=cfg.rl_device, headless=cfg.headless)
    env.reset()
    env.close()

if __name__ == "__main__":
    main()