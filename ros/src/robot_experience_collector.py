#!/usr/bin/env python
import torch
import numpy as np
import rospy
import rospkg
from storm_ros.srv import ReachGoal, ReachGoalResponse
from storm_ros.msg import GoalMsg
from storm_kit.learning.experience_collector import ExperienceCollector
from storm_kit.learning.replay_buffers import RobotBuffer

class RobotExperienceCollector():
    def __init__(self, cfg):
        self.cfg = cfg
        self.episode_length = cfg['env']['episodeLength']
        self.n_dofs = cfg.task.model.n_dofs

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.joint_names = rospy.get_param('~robot_joint_names', ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        
        self.device = torch.device('cuda', 0)






        print('Waiting for service')
        rospy.wait_for_service('mpc')
        print('Service running')

    def collect_experience(self, 
                           num_episodes:int,
                           update_buffer:bool=True):
        if update_buffer:
            self.buffer = RobotBuffer(capacity=num_episodes*self.episode_length, n_dofs=self.n_dofs)

        try:
            mpc_service = rospy.ServiceProxy('mpc', ReachGoal, persistent=True)
            input('Press any key to initiate reset')
            _ = self.send_reset_command(mpc_service)

            for ep_num in range(num_episodes):
                input('Press any key to collect episode')
                ret = self.collect_episode(mpc_service)
                if update_buffer:
                    self.add_to_buffer(ret)
                input('Episode {} done. Press any key to initiate reset'.format(ep_num+1))
                _ = self.send_reset_command(mpc_service)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)       

    def collect_episode(self, mpc_service):
        goal_msg = GoalMsg()
        goal_msg.MODE_FLAG = 0
        goal_msg.ee_goal.position.x = 0.1
        goal_msg.ee_goal.position.y = 0.0
        goal_msg.ee_goal.position.z = 0.5
        goal_msg.ee_goal.orientation.x = 0.707
        goal_msg.ee_goal.orientation.y = 0.0
        goal_msg.ee_goal.orientation.z = 0.0
        goal_msg.ee_goal.orientation.w = 0.707
        goal_msg.joint_goal.name=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        goal_msg.joint_goal.position = [0.0, -0.7853, 0.0, -2.3561, 0.0, 1.5707, 0.7853]
        goal_msg.joint_goal.velocity = [0.0] * 7
        goal_msg.joint_goal.effort = [0.0] * 7

        service_req = ReachGoal()
        service_req.goal = goal_msg
        
        ret = mpc_service(goal_msg)

        return ret

    def send_reset_command(self, mpc_service):
            
        goal_msg = GoalMsg()
        goal_msg.MODE_FLAG = 1
        goal_msg.ee_goal.position.x = 0.0
        goal_msg.ee_goal.position.y = 0.0
        goal_msg.ee_goal.position.z = 0.0
        goal_msg.ee_goal.orientation.x = 0.0
        goal_msg.ee_goal.orientation.y = 0.0
        goal_msg.ee_goal.orientation.z = 0.0
        goal_msg.ee_goal.orientation.w = 0.0
        goal_msg.joint_goal.name=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        goal_msg.joint_goal.position = [0.0, -0.7853, 0.0, -2.3561, 0.0, 1.5707, 0.7853]
        goal_msg.joint_goal.velocity = [0.0] * 7
        goal_msg.joint_goal.effort = [0.0] * 7




        service_req = ReachGoal()
        service_req.goal = goal_msg
        
        ret = mpc_service(goal_msg)

        return ret

    def add_to_buffer(self, ret):
        pass


if __name__ == "__main__":

    experience_collector = RobotExperienceCollector()
    experience_collector.collect_experience(num_episodes=10)