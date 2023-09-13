#!/usr/bin/env python

#General imports
import os
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#ROS Imports
import rospy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
import rospkg
# from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
# from std_msgs.msg import String, Header

#STORM imports
from storm_kit.mpc.rollout import ArmReacher
from storm_kit.learning.policies import MPCPolicy, JointControlWrapper
from storm_kit.learning.learning_utils import dict_to_device

class MPCReacherNode():
    def __init__(self) -> None:
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.joint_names = rospy.get_param('~robot_joint_names', None)
        self.save_data = rospy.get_param('~save_data', False)
        
        initialize(config_path="../../../content/configs/gym", job_name="mpc")
        self.config = compose(config_name="config", overrides=["task=FrankaReacherRealRobot"])
        self.control_dt = self.config.task.rollout.control_dt
        self.n_dofs = self.config.task.rollout.n_dofs
        self.device = self.config.rl_device

        #STORM Initialization
        obs_dim = 1
        act_dim = 1
        self.policy = MPCPolicy(
            obs_dim=obs_dim, act_dim=act_dim, 
            config=self.config.task.mpc, rollout_cls = ArmReacher, 
            device=self.device)
        self.policy = JointControlWrapper(config=self.config.task.mpc, policy=self.policy, device=self.device)

        # if self.save_data:
        #     buffer = RobotBuffer(capacity=1e-6)


        #buffers for different messages
        self.mpc_command = JointState()
        self.mpc_command.name = self.joint_names
        # self.gripper_state = JointState()
        # self.robot_state = JointState()
        self.command_header = None
        self.gripper_state = {
            'q_pos': torch.zeros(1,2),
            'q_vel': torch.zeros(1,2),
            'q_acc': torch.zeros(1,2)}
        
        self.robot_state = {
            'q_pos': torch.zeros(1,7, device=self.device),
            'q_vel': torch.zeros(1,7, device=self.device),
            'q_acc': torch.zeros(1,7, device=self.device)}


        self.robot_state_tensor =  torch.cat((
            self.robot_state['q_pos'],
            self.robot_state['q_vel'],
            self.robot_state['q_acc']
        ), dim=-1)

        self.policy_input = {'states': self.robot_state}
        self.goal_dict = {}
        self.ee_goal_pos = None
        self.ee_goal_quat = None

        #ROS Initialization
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        self.ee_goal_sub = rospy.Subscriber(self.ee_goal_topic, PoseStamped, self.ee_goal_callback, queue_size=1)
        self.control_freq = float(1.0/self.control_dt)
        self.rate = rospy.Rate(self.control_freq)

        self.state_sub_on = False
        self.goal_sub_on = False
        self.tstep = 0
        self.tstep_tensor = torch.as_tensor([self.tstep], device=self.config.rl_device).unsqueeze(0)

        self.start_t = None
        self.first_iter = True

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
        self.robot_state['q_pos'] = torch.tensor([msg.position], device=self.device)
        self.robot_state['q_vel'] = torch.tensor([msg.velocity], device=self.device)
        # self.robot_state['q_acc'] = torch.zeros_like(self.robot_state['q_vel'])
        # self.robot_state_tensor = torch.cat((
        #     self.robot_state['q_pos'],
        #     self.robot_state['q_vel'],
        #     self.robot_state['q_acc']
        # ), dim=-1)



    def ee_goal_callback(self, msg):
        self.goal_sub_on = True
        self.new_ee_goal = True
        self.ee_goal_pos = torch.tensor([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z])
        self.ee_goal_quat = torch.tensor([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z])

        goal = torch.cat((self.ee_goal_pos, self.ee_goal_quat), dim=-1).unsqueeze(0)
        self.goal_dict['ee_goal'] = goal


    def control_loop(self):
        while not rospy.is_shutdown():
            #only do something if state and goal have been received
            if self.state_sub_on and self.goal_sub_on:
                #check if goal was updated
                if self.new_ee_goal:
                    param_dict = {'goal_dict': dict_to_device(self.goal_dict, device=self.device)}
                    self.policy.update_rollout_params(param_dict)
                    self.new_ee_goal = False

        
                # for k in self.robot_state.keys():
                #     self.policy_input['states'][k] = self.robot_state[k].clone().to(self.device)
                                
                self.tstep_tensor[0,0] = self.tstep
                self.policy_input['states']['tstep'] = self.tstep_tensor
                self.policy_input['obs'] = self.robot_state_tensor.to(self.device)

                # for k in self.policy_input['states'].keys():
                #     print('in node', k, self.policy_input['states'][k].device)



                #get mpc command
                command_tensor, policy_info = self.policy.get_action(self.policy_input)

                #publish mpc command
                self.mpc_command.header = self.command_header
                self.mpc_command.header.stamp = rospy.Time.now()
                self.mpc_command.position = command_tensor[0][0:7].cpu().numpy()
                self.mpc_command.velocity = command_tensor[0][7:14].cpu().numpy()
                self.mpc_command.effort =  command_tensor[0][14:21].cpu().numpy()

                # self.mpc_command.position = command['q_des'][0].cpu().numpy()
                # self.mpc_command.velocity = command['qd_des'][0].cpu().numpy()
                # self.mpc_command.effort =  command['qdd_des'][0].cpu().numpy()
                self.command_pub.publish(self.mpc_command)

                #update tstep
                if self.tstep == 0:
                    rospy.loginfo('[MPCPoseReacher]: Controller running')
                    self.start_t = rospy.get_time()
                self.tstep = rospy.get_time() - self.start_t

            else:
                if (not self.state_sub_on) and (self.first_iter):
                    rospy.loginfo('[MPCPoseReacher]: Waiting for robot state.')
                if (not self.goal_sub_on) and (self.first_iter):
                    rospy.loginfo('[MPCPoseReacher]: Waiting for ee goal.')
            
            self.first_iter = False
            self.rate.sleep()
    
    def close(self):
        self.command_pub.unregister()
        self.state_sub.unregister()
        self.ee_goal_sub.unregister()


if __name__ == "__main__":
    rospy.init_node("mpc_reacher_node", anonymous=True, disable_signals=True)    
    torch.set_default_dtype(torch.float32)

    mpc_node = MPCReacherNode()

    try:
        mpc_node.control_loop()
    except KeyboardInterrupt:
        print('Exiting')
        mpc_node.close()