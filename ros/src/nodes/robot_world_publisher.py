#!/usr/bin/env python
"""
Node that publishes world and robot collision model.
"""
import rospy
from sensor_msgs.msg import JointState
import torch

class RobotWorldPublisher():
    def __init__(self) -> None:
        pass
        self.rate = rospy.Rate(500)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        self.state_sub_on = False

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