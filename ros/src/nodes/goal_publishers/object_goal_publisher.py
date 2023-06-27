#!/usr/bin/env python
from copy import deepcopy
import numpy as np
import os
import torch

import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import JointState
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *
import rospkg
import tf2_ros

from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.coordinate_transform import matrix_to_quaternion


class ObjectGoalPub():
    def __init__(self):
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.goal_pub_freq = rospy.get_param('~goal_pub_freq', 10)
        self.fixed_frame = rospy.get_param('~fixed_frame', 'base_link')
        self.ee_frame = rospy.get_param('~ee_frame', 'tray_link')
        self.object_frame = rospy.get_param('~object_frame', 'object')
        self.robot_urdf = os.path.join(self.storm_path, rospy.get_param(
            '~robot_urdf', 'content/assets/urdf/franka_description/franka_panda_no_gripper.urdf'))
        self.z_offset = float(rospy.get_param('~z_offset', '0.05'))

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.br = tf2_ros.TransformBroadcaster()

        #ROS Initialization
        self.ee_goal = PoseStamped()
        self.gripper_state = JointState()
        self.robot_state = JointState()

        self.ee_goal_pub = rospy.Publisher(self.ee_goal_topic, PoseStamped, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)

        self.robot_model = DifferentiableRobotModel(self.robot_urdf, None)

        self.rate = rospy.Rate(self.goal_pub_freq)
        self.state_received = False
        
        while not self.state_received:
            pass

        #we set self.ee_goal to the initial robot pose
        self.update_ee_goal_to_current()

    def goal_pub_loop(self):
        while not rospy.is_shutdown():
            try:
                trans = self.tfBuffer.lookup_transform(self.fixed_frame, self.object_frame, rospy.Time(0))
                self.ee_goal.pose.position.x = trans.transform.translation.x 
                self.ee_goal.pose.position.y = trans.transform.translation.y
                self.ee_goal.pose.position.z = trans.transform.translation.z
                self.ee_goal.pose.orientation.x = 0.707
                self.ee_goal.pose.orientation.y = 0.707
                self.ee_goal.pose.orientation.z = 0.0
                self.ee_goal.pose.orientation.w = 0.0 #0.707
                print('Setting goal to object pose')
    
            except Exception as e:
                print(e)
                print('Setting goal to current EE pose')
                self.update_ee_goal_to_current()



            transform_msg = TransformStamped()
            transform_msg.header.stamp = rospy.Time().now()#self.image_msg.header.stamp
            transform_msg.header.frame_id = self.fixed_frame
            transform_msg.child_frame_id = "ee_goal"
            transform_msg.transform.translation.x = self.ee_goal.pose.position.x
            transform_msg.transform.translation.y = self.ee_goal.pose.position.y
            transform_msg.transform.translation.z = self.ee_goal.pose.position.z + self.z_offset

            transform_msg.transform.rotation.x = self.ee_goal.pose.orientation.x
            transform_msg.transform.rotation.y = self.ee_goal.pose.orientation.y
            transform_msg.transform.rotation.z = self.ee_goal.pose.orientation.z
            transform_msg.transform.rotation.w = self.ee_goal.pose.orientation.w
            self.br.sendTransform(transform_msg)

            self.ee_goal_pub.publish(self.ee_goal)
            self.rate.sleep()
    
    def robot_state_callback(self, msg):
        self.state_received = True
        # save gripper state
        # self.gripper_state.header = msg.header
        # self.gripper_state.position = msg.position[0:2]
        # self.gripper_state.velocity = msg.velocity[0:2]
        # self.gripper_state.effort = msg.effort[0:2]

        #save robot state
        self.robot_state.header = msg.header
        self.robot_state.position = msg.position#[2:]
        self.robot_state.velocity = msg.velocity#[2:]
        self.robot_state.effort = msg.effort#[2:]


    def update_ee_goal_to_current(self):
        q_robot = torch.as_tensor(self.robot_state.position).unsqueeze(0)
        qd_robot = torch.as_tensor(self.robot_state.velocity).unsqueeze(0)
        # q_gripper = torch.as_tensor(self.gripper_state.position, **self.tensor_args).unsqueeze(0)
        # qd_gripper = torch.as_tensor(self.gripper_state.velocity, **self.tensor_args).unsqueeze(0)
        # q = torch.cat((q_robot, q_gripper), dim=-1)
        # qd = torch.cat((qd_robot, qd_gripper), dim=-1)


        self.curr_ee_pos, self.curr_ee_rot = self.robot_model.compute_forward_kinematics(
            q_robot, qd_robot, link_name=self.ee_frame)
        self.curr_ee_quat = matrix_to_quaternion(self.curr_ee_rot)
        # self.curr_ee_quat = self.curr_ee_quat / torch.norm(self.curr_ee_quat) #normalize quaternion

        #convert to pose stamped message
        self.ee_goal.header.stamp = rospy.Time.now()
        self.ee_goal.pose.position.x = self.curr_ee_pos[0][0].item() 
        self.ee_goal.pose.position.y = self.curr_ee_pos[0][1].item() 
        self.ee_goal.pose.position.z = self.curr_ee_pos[0][2].item() 
        self.ee_goal.pose.orientation.w = self.curr_ee_quat[0][0].item() 
        self.ee_goal.pose.orientation.x = self.curr_ee_quat[0][1].item() 
        self.ee_goal.pose.orientation.y = self.curr_ee_quat[0][2].item() 
        self.ee_goal.pose.orientation.z = self.curr_ee_quat[0][3].item()
    
    def close(self):
        self.ee_goal_pub.unregister()
        self.state_sub.unregister()

if __name__ == "__main__":
    rospy.init_node("object_goal_node", anonymous=True, disable_signals=True)    

    goal_node = ObjectGoalPub()

    input('Press enter to start tracking')
    try:
        goal_node.goal_pub_loop()
    except KeyboardInterrupt:
        print('Exiting')
        goal_node.close()
