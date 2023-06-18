#!/usr/bin/env python
from copy import deepcopy
import numpy as np
import os
import torch

import rospy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *
import rospkg

from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.coordinate_transform import matrix_to_quaternion


class InteractiveMarkerGoalPub():
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
        

        #ROS Initialization
        self.ee_goal = PoseStamped()
        self.gripper_state = JointState()
        self.robot_state = JointState()

        self.ee_goal_pub = rospy.Publisher(self.ee_goal_topic, PoseStamped, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)

        #STORM Related
        self.tensor_args = {'device': 'cpu', 'dtype': torch.float32}
        self.robot_model = DifferentiableRobotModel(self.robot_urdf, None, 
                            tensor_args=self.tensor_args)
            
        #Buffers

        self.rate = rospy.Rate(self.goal_pub_freq)
        self.state_received = False
        while not self.state_received:
            pass
        #we set self.ee_goal to the initial robot pose
        self.update_ee_goal_to_current()

        self.setup_interactive_marker_server()




    def setup_interactive_marker_server(self):
        # create an interactive marker server on the topic namespace simple_marker
        self.server = InteractiveMarkerServer("goal_marker")

        # create an interactive marker for our server
        self.int_marker = InteractiveMarker()
        self.int_marker.header.frame_id = self.fixed_frame #"panda_link0"
        self.int_marker.name = "goal_marker"
        self.int_marker.description = "End-effector Goal"
        self.int_marker.scale = 0.2
        # self.int_marker.pose.position = self.curr_goal_ros.pose.position
        self.int_marker.pose = self.ee_goal.pose


        # create a grey box marker
        self.box_marker = Marker()
        self.box_marker.type = Marker.CUBE
        self.box_marker.scale.x = 0.1
        self.box_marker.scale.y = 0.1
        self.box_marker.scale.z = 0.1
        self.box_marker.color.r = 0.0
        self.box_marker.color.g = 0.5
        self.box_marker.color.b = 0.5
        self.box_marker.color.a = 1.0

        # create a non-interactive control which contains the box
        self.box_control = InteractiveMarkerControl()
        self.box_control.always_visible = True
        self.box_control.markers.append(self.box_marker)

        # add the control to the interactive marker
        self.int_marker.controls.append(self.box_control)

        # create a control which will move the box
        # this control does not contain any markers,
        # which will cause RViz to insert two arrows
        self.move_x_control = InteractiveMarkerControl()
        self.move_x_control.name = "move_x"
        self.move_x_control.orientation.w = 1
        self.move_x_control.orientation.x = 1
        self.move_x_control.orientation.y = 0
        self.move_x_control.orientation.z = 0   
        self.move_x_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        # add the control to the interactive marker
        self.int_marker.controls.append(self.move_x_control)

        self.move_y_control = InteractiveMarkerControl()
        self.move_y_control.name = "move_y"
        self.move_y_control.orientation.w = 1
        self.move_y_control.orientation.x = 0
        self.move_y_control.orientation.y = 0
        self.move_y_control.orientation.z = 1                
        self.move_y_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        self.int_marker.controls.append(self.move_y_control)

        self.move_z_control = InteractiveMarkerControl()
        self.move_z_control.name = "move_z"
        self.move_z_control.orientation.w = 1
        self.move_z_control.orientation.x = 0
        self.move_z_control.orientation.y = 1
        self.move_z_control.orientation.z = 0
        self.move_z_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS

        self.int_marker.controls.append(self.move_z_control)

        # create controls for rotation
        self.rotate_x_control = InteractiveMarkerControl()
        self.rotate_x_control.orientation.w = 1
        self.rotate_x_control.orientation.x = 1
        self.rotate_x_control.orientation.y = 0
        self.rotate_x_control.orientation.z = 0
        self.rotate_x_control.name = "rotate_x"
        self.rotate_x_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        # if fixed:
            # control.orientation_mode = InteractiveMarkerControl.FIXED
        self.int_marker.controls.append(self.rotate_x_control)

        self.rotate_y_control = InteractiveMarkerControl()
        self.rotate_y_control.orientation.w = 1
        self.rotate_y_control.orientation.x = 0
        self.rotate_y_control.orientation.y = 1
        self.rotate_y_control.orientation.z = 0
        self.rotate_y_control.name = "rotate_y"
        self.rotate_y_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        # if fixed:
            # control.orientation_mode = InteractiveMarkerControl.FIXED
        self.int_marker.controls.append(self.rotate_y_control)

        self.rotate_z_control = InteractiveMarkerControl()
        self.rotate_z_control.orientation.w = 1
        self.rotate_z_control.orientation.x = 0
        self.rotate_z_control.orientation.y = 0
        self.rotate_z_control.orientation.z = 1
        self.rotate_z_control.name = "rotate_z"
        self.rotate_z_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        # if fixed:
            # control.orientation_mode = InteractiveMarkerControl.FIXED
        self.int_marker.controls.append(self.rotate_z_control)

        # add the interactive marker to our collection &
        # tell the server to call marker_callback() when feedback arrives for it
        self.server.insert(self.int_marker, self.marker_callback)

        # 'commit' changes and send to all clients
        self.server.applyChanges()

    def marker_callback(self, msg):
        self.ee_goal.header = msg.header
        self.ee_goal.pose = deepcopy(msg.pose)

    def goal_pub_loop(self):
        while not rospy.is_shutdown():
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
        q_robot = torch.as_tensor(self.robot_state.position, **self.tensor_args).unsqueeze(0)
        qd_robot = torch.as_tensor(self.robot_state.velocity, **self.tensor_args).unsqueeze(0)
        # q_gripper = torch.as_tensor(self.gripper_state.position, **self.tensor_args).unsqueeze(0)
        # qd_gripper = torch.as_tensor(self.gripper_state.velocity, **self.tensor_args).unsqueeze(0)
        # q = torch.cat((q_robot, q_gripper), dim=-1)
        # qd = torch.cat((qd_robot, qd_gripper), dim=-1)


        curr_ee_pos, curr_ee_rot = self.robot_model.compute_forward_kinematics(
            q_robot, qd_robot, link_name=self.ee_frame)
        curr_ee_quat = matrix_to_quaternion(curr_ee_rot)
        # self.curr_ee_quat = self.curr_ee_quat / torch.norm(self.curr_ee_quat) #normalize quaternion


        #convert to pose stamped message
        self.ee_goal.header.stamp = rospy.Time.now()
        self.ee_goal.pose.position.x = curr_ee_pos[0][0].item() 
        self.ee_goal.pose.position.y = curr_ee_pos[0][1].item() 
        self.ee_goal.pose.position.z = curr_ee_pos[0][2].item() 
        self.ee_goal.pose.orientation.w = curr_ee_quat[0][0].item() 
        self.ee_goal.pose.orientation.x = curr_ee_quat[0][1].item() 
        self.ee_goal.pose.orientation.y = curr_ee_quat[0][2].item() 
        self.ee_goal.pose.orientation.z = curr_ee_quat[0][3].item()
    
    def close(self):
        self.ee_goal_pub.unregister()
        self.state_sub.unregister()

if __name__ == "__main__":
    rospy.init_node("interactive_marker_goal_node", anonymous=True, disable_signals=True)    

    goal_node = InteractiveMarkerGoalPub()

    try:
        goal_node.goal_pub_loop()
    except KeyboardInterrupt:
        print('Exiting')
        goal_node.close()
