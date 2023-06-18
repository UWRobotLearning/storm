#!/usr/bin/env python

from copy import deepcopy
import numpy as np
import os
import torch
import yaml

import rospy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *
import tf_conversions
import tf2_ros
import rospkg

from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.coordinate_transform import matrix_to_quaternion, quaternion_to_matrix


class SequenceGoalPublisher():
    def __init__(self):

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.goal_pub_freq = rospy.get_param('~goal_pub_freq', 10)
        self.fixed_frame = rospy.get_param('~fixed_frame', 'base_link')
        self.robot_urdf = os.path.join(self.storm_path, rospy.get_param('~robot_urdf', 'content/assets/urdf/franka_description/franka_panda_tray.urdf'))
        self.ee_frame = rospy.get_param('~ee_frame', 'tray_link')
        self.goal_file = os.path.join(self.storm_path,rospy.get_param('goal_list_file', 'storm_ros/src/nodes/goal_publishers/left_right_goal.yaml'))
        self.br = tf2_ros.TransformBroadcaster()

        
        with open(self.goal_file, "r") as f:
            try:
                self.goal_list = yaml.safe_load(f)['goal_list']
            except yaml.YAMLError as exc:
                print(exc)
        self.num_goals = len(self.goal_list)
        self.curr_goal_idx = 0
        self.goal_update_secs = 20

        #ROS Initialization
        self.ee_goal = PoseStamped()
        self.gripper_state = JointState()
        self.robot_state = JointState()

        self.ee_goal_pub = rospy.Publisher(self.ee_goal_topic, PoseStamped, queue_size=1, tcp_nodelay=True, latch=False)
        self.goal_vis_pub = rospy.Publisher('goal_marker', MarkerArray, queue_size=1)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)

        #STORM Related
        self.tensor_args = {'device': 'cpu', 'dtype': torch.float32}
        self.robot_model = DifferentiableRobotModel(self.robot_urdf, None, 
                            tensor_args=self.tensor_args)
            
        #Buffers
        self.tstep = 0
        self.last_goal_tstep = 0.0
        self.start_t = None
        self.rate = rospy.Rate(self.goal_pub_freq)
        self.state_received = False
        while not self.state_received:
            pass
        #we set self.ee_goal to the initial robot pose
        self.update_ee_goal_to_current()
        self.setup_goal_marker()

    def marker_callback(self, msg):
        self.ee_goal.header = msg.header
        self.ee_goal.pose = deepcopy(msg.pose)

    def goal_pub_loop(self):
        while not rospy.is_shutdown():
            curr_goal = self.goal_list[self.curr_goal_idx]
            self.ee_goal.pose.position.x = curr_goal[0]
            self.ee_goal.pose.position.y = curr_goal[1]
            self.ee_goal.pose.position.z = curr_goal[2]
            self.ee_goal.pose.orientation.x = 0.707388
            self.ee_goal.pose.orientation.y = 0.706825
            self.ee_goal.pose.orientation.z = -0.0005629
            self.ee_goal.pose.orientation.w = 0.0005633

            self.ee_goal_pub.publish(self.ee_goal)

            t = geometry_msgs.msg.TransformStamped()

            t.header.stamp = rospy.Time.now()
            t.header.frame_id = self.fixed_frame
            t.child_frame_id = 'ee_goal'
            t.transform.translation.x = self.ee_goal.pose.position.x
            t.transform.translation.y = self.ee_goal.pose.position.y
            t.transform.translation.z = self.ee_goal.pose.position.z
            t.transform.rotation.x = self.ee_goal.pose.orientation.x
            t.transform.rotation.y = self.ee_goal.pose.orientation.y
            t.transform.rotation.z = self.ee_goal.pose.orientation.z
            t.transform.rotation.w = self.ee_goal.pose.orientation.w

            self.br.sendTransform(t)

            #update goal idx based on some criterion
            if self.tstep == 0:
                self.start_t = rospy.get_time()
            
            self.tstep = rospy.get_time() - self.start_t
            
            if (self.tstep - self.last_goal_tstep) >= self.goal_update_secs :
                print('updating goal')
                self.curr_goal_idx = (self.curr_goal_idx + 1) % self.num_goals
                self.last_goal_tstep = deepcopy(self.tstep)  

            self.update_goal_marker_and_publish()

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


    def get_ee_pose(self):
        q_robot = torch.as_tensor(self.robot_state.position, **self.tensor_args).unsqueeze(0)
        qd_robot = torch.as_tensor(self.robot_state.velocity, **self.tensor_args).unsqueeze(0)


        curr_ee_pos, curr_ee_rot = self.robot_model.compute_forward_kinematics(
            q_robot, qd_robot, link_name=self.ee_frame)
        curr_ee_quat = matrix_to_quaternion(curr_ee_rot)
        return curr_ee_pos, curr_ee_quat

    def update_ee_goal_to_current(self):
        q_robot = torch.as_tensor(self.robot_state.position, **self.tensor_args).unsqueeze(0)
        qd_robot = torch.as_tensor(self.robot_state.velocity, **self.tensor_args).unsqueeze(0)

        curr_ee_pos, curr_ee_rot = self.robot_model.compute_forward_kinematics(
            q_robot, qd_robot, link_name=self.ee_frame)
        curr_ee_quat = matrix_to_quaternion(curr_ee_rot)

        #convert to pose stamped message
        self.ee_goal.header.stamp = rospy.Time.now()
        self.ee_goal.pose.position.x = curr_ee_pos[0][0].item() 
        self.ee_goal.pose.position.y = curr_ee_pos[0][1].item() 
        self.ee_goal.pose.position.z = curr_ee_pos[0][2].item() 
        self.ee_goal.pose.orientation.w = curr_ee_quat[0][0].item() 
        self.ee_goal.pose.orientation.x = curr_ee_quat[0][1].item() 
        self.ee_goal.pose.orientation.y = curr_ee_quat[0][2].item() 
        self.ee_goal.pose.orientation.z = curr_ee_quat[0][3].item()
    
    def setup_goal_marker(self):
        self.goal_marker = MarkerArray()

        self.marker_center = Marker()
        self.marker_center.type = Marker.SPHERE
        self.marker_center.action = Marker.ADD
        self.marker_center.id = 0
        self.marker_center.header.frame_id = self.fixed_frame
        self.marker_center.pose = self.ee_goal.pose
        self.marker_center.scale.x = 0.05
        self.marker_center.scale.y = 0.05
        self.marker_center.scale.z = 0.05
        self.marker_center.color.a = 1.0 
        self.marker_center.color.r = 1.0
        self.marker_center.color.g = 1.0
        self.marker_center.color.b = 1.0

        self.marker_x = Marker()
        self.marker_x.type = Marker.ARROW
        self.marker_x.action = Marker.ADD
        self.marker_x.id = 1
        self.marker_x.header.frame_id = self.fixed_frame
        self.marker_x.pose.position = self.ee_goal.pose.position
        self.marker_x.pose.orientation.x = self.ee_goal.pose.orientation.x
        self.marker_x.pose.orientation.y = 0.0
        self.marker_x.pose.orientation.z = 0.0
        self.marker_x.pose.orientation.w = 1.0 #self.ee_goal.pose.orientation.w
        self.marker_x.scale.x = 0.1
        self.marker_x.scale.y = 0.01
        self.marker_x.scale.z = 0.01
        self.marker_x.color.a = 1.0 
        self.marker_x.color.r = 1.0
        self.marker_x.color.g = 0.0
        self.marker_x.color.b = 0.0


        self.marker_y = Marker()
        self.marker_y.type = Marker.ARROW
        self.marker_y.action = Marker.ADD
        self.marker_y.id = 2
        self.marker_y.header.frame_id = self.fixed_frame
        self.marker_y.pose.position = self.ee_goal.pose.position
        self.marker_y.pose.orientation.x = 0.0 
        self.marker_y.pose.orientation.y = self.ee_goal.pose.orientation.y
        self.marker_y.pose.orientation.z = 0.0
        self.marker_y.pose.orientation.w = 1.0 #self.ee_goal.pose.orientation.w
        self.marker_y.scale.x = 0.1
        self.marker_y.scale.y = 0.01
        self.marker_y.scale.z = 0.01
        self.marker_y.color.a = 1.0 
        self.marker_y.color.r = 0.0
        self.marker_y.color.g = 1.0
        self.marker_y.color.b = 0.0

        self.marker_z = Marker()
        self.marker_z.type = Marker.ARROW
        self.marker_z.action = Marker.ADD
        self.marker_z.id = 3
        self.marker_z.header.frame_id = self.fixed_frame
        self.marker_z.pose.position = self.ee_goal.pose.position
        self.marker_z.pose.orientation.x = 0.0 
        self.marker_z.pose.orientation.y = 0.0
        self.marker_z.pose.orientation.z = self.ee_goal.pose.orientation.z
        self.marker_z.pose.orientation.w = 1.0 #self.ee_goal.pose.orientation.w
        self.marker_z.scale.x = 0.1
        self.marker_z.scale.y = 0.01
        self.marker_z.scale.z = 0.01
        self.marker_z.color.a = 1.0 
        self.marker_z.color.r = 0.0
        self.marker_z.color.g = 0.0
        self.marker_z.color.b = 1.0


        self.goal_marker.markers.append(self.marker_center)
        # self.goal_marker.markers.append(self.marker_x)
        # self.goal_marker.markers.append(self.marker_y)
        # self.goal_marker.markers.append(self.marker_z)


    def update_goal_marker_and_publish(self):
        self.marker_center.header.stamp = self.ee_goal.header.stamp
        self.marker_center.pose = self.ee_goal.pose
        

        self.marker_x.header.stamp = self.ee_goal.header.stamp
        self.marker_x.pose.position = self.ee_goal.pose.position
        self.marker_x.pose.orientation.x = self.ee_goal.pose.orientation.x
        self.marker_x.pose.orientation.y = 0.0
        self.marker_x.pose.orientation.z = 0.0
        self.marker_x.pose.orientation.w = self.ee_goal.pose.orientation.w

        self.marker_y.header.stamp = self.ee_goal.header.stamp
        self.marker_y.pose.position = self.ee_goal.pose.position
        self.marker_y.pose.orientation.x = 0.0
        self.marker_y.pose.orientation.y = self.ee_goal.pose.orientation.y
        self.marker_y.pose.orientation.z = 0.0
        self.marker_y.pose.orientation.w = 0.0 #self.ee_goal.pose.orientation.w

        self.marker_z.header.stamp = self.ee_goal.header.stamp
        self.marker_z.pose.position = self.ee_goal.pose.position
        self.marker_z.pose.orientation.x = 0.0
        self.marker_z.pose.orientation.y = 0.0
        self.marker_z.pose.orientation.z = self.ee_goal.pose.orientation.z
        self.marker_z.pose.orientation.w = self.ee_goal.pose.orientation.w
        
        
        self.goal_vis_pub.publish(self.goal_marker)





    def close(self):
        self.ee_goal_pub.unregister()
        self.state_sub.unregister()

if __name__ == "__main__":
    rospy.init_node("interactive_marker_goal_node", anonymous=True, disable_signals=True)    

    goal_node = SequenceGoalPublisher()

    try:
        goal_node.goal_pub_loop()
    except KeyboardInterrupt:
        print('Exiting')
        goal_node.close()
