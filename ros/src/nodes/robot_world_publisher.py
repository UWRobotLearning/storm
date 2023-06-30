#!/usr/bin/env python
"""
Node that publishes world and robot collision model.
"""
import rospy
from sensor_msgs.msg import JointState
import torch

from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.geom.robot import RobotSphereCollision

class RobotWorldPublisher():
    def __init__(self) -> None:
        pass
        self.rate = rospy.Rate(500)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        self.state_sub_on = False
        self.urdf_path = join_path(get_assets_path(), robot_collision_config['urdf_path'])
        self.link_names = robot_collision_config['link_names']

        self.robot_model = DifferentiableRobotModel(urdf_path)
        self.collision_model = RobotSphereCollision(robot_collision_config)
        self.collision_model.build_batch_features(batch_size=1, clone_pose=True, clone_objs=True)

        self.robot_state = None

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


    def load_world_model(self):
        pass

    def load_robot_model(self):
        pass

    def pub_loop(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def close(self):
        self.state_sub.unregister()

if __name__ == "__main__":
    rospy.init_node("robot_world_publisher", anonymous=True, disable_signals=True)    
    torch.set_default_dtype(torch.float32)

    obj = RobotWorldPublisher()

    try:
        obj.pub_loop()
    except KeyboardInterrupt:
        obj.close()