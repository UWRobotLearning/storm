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
# from storm_kit.mpc.task.reacher_task import ReacherTask
from storm_kit.rl.policies import MPCPolicy


class MPCReacherNode():
    def __init__(self) -> None:
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.world_description = os.path.join(self.storm_path, rospy.get_param('~world_description', '../content/configs/gym/collision_wall_of_boxes.yml'))
        self.robot_coll_description = os.path.join(self.storm_path, rospy.get_param('~robot_coll_description', '../content/configs/robot/franka_real_robot.yml'))
        self.mpc_config = os.path.join(self.storm_path, rospy.get_param('~mpc_config', '../content/configs/mpc/franka_real_robot_reacher.yml'))
        self.control_dt = rospy.get_param('~control_dt', 0.02)

        self.joint_names = rospy.get_param('~robot_joint_names', None)
        
        self.device = torch.device('cuda', 0)

        initialize(config_path="../../../content/configs/gym", job_name="mpc")
        self.config = compose(config_name="config", overrides=["task=FrankaReacherRealRobot"])

        #STORM Initialization
        # self.policy = ReacherTask(self.mpc_config, self.robot_coll_description, self.world_description, self.tensor_args)
        obs_dim = 1
        act_dim = 1
        self.policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=self.config.mpc, device=self.device)


        #buffers for different messages
        self.mpc_command = JointState()
        self.mpc_command.name = self.joint_names
        self.mpc_command.effort = np.zeros(7)
        # self.gripper_state = JointState()
        # self.robot_state = JointState()
        self.command_header = None
        self.gripper_state = {
            'position': torch.zeros(2),
            'velocity': torch.zeros(2),
            'acceleration': torch.zeros(2)}
        self.robot_state = {
            'position': torch.zeros(7),
            'velocity': torch.zeros(7),
            'acceleration': torch.zeros(7)}
        self.obs_dict = {}
        self.robot_state_tensor = None
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
        self.robot_state['position'] = torch.tensor(msg.position)
        self.robot_state['velocity'] = torch.tensor(msg.velocity)
        self.robot_state['acceleration'] = torch.zeros_like(self.robot_state['velocity'])
        self.robot_state_tensor = torch.cat((
            self.robot_state['position'],
            self.robot_state['velocity'],
            self.robot_state['acceleration']
        )).unsqueeze(0)
        self.obs_dict = {'states':self.robot_state_tensor}

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

    def control_loop(self):
        while not rospy.is_shutdown():
            #only do something if state and goal have been received
            if self.state_sub_on and self.goal_sub_on:
                #check if goal was updated
                if self.new_ee_goal:
                    # self.policy.update_params(goal_ee_pos = self.ee_goal_pos,
                    #     goal_ee_quat = self.ee_goal_quat)
                    goal = torch.cat((self.ee_goal_pos, self.ee_goal_quat), dim=-1).unsqueeze(0)
                    self.policy.update_goal(goal)
                    self.new_ee_goal = False

                self.obs_dict['states'] = torch.cat(
                    (self.obs_dict['states'],
                     torch.as_tensor([self.tstep]).unsqueeze(0)),
                     dim=-1                  
                    )

                #get mpc command
                command = self.policy.get_action(obs_dict=self.obs_dict) #, control_dt=self.control_dt, WAIT=True)

                #publish mpc command
                self.mpc_command.header = self.command_header
                self.mpc_command.header.stamp = rospy.Time.now()
                self.mpc_command.position = command['q_des'][0].cpu().numpy()
                self.mpc_command.velocity = command['qd_des'][0].cpu().numpy()
                self.mpc_command.effort =  command['qdd_des'][0].cpu().numpy()

                print(self.mpc_command)

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