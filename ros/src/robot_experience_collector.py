#!/usr/bin/env python
import numpy as np
import rospy

from storm_ros.srv import ReachGoal, ReachGoalResponse
from storm_ros.msg import GoalMsg


class RobotExperienceCollector():
    def __init__(self):
        print('Waiting for service')
        rospy.wait_for_service('mpc')
        print('Service running')

    def collect_experience(self, num_episodes):
        try:
            mpc_service = rospy.ServiceProxy('mpc', ReachGoal, persistent=True)
            input('Press any key to initiate reset')
            ret = self.send_reset_command(mpc_service)

            for ep_num in range(num_episodes):
                input('Press any key to collect episode')
                ret = self.send_ee_goal_command(mpc_service)
                input('Episode {} done. Press any key to initiate reset'.format(ep_num+1))
                ret = self.send_reset_command(mpc_service)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)       

    def send_ee_goal_command(self, mpc_service):
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


if __name__ == "__main__":

    experience_collector = RobotExperienceCollector()
    experience_collector.collect_experience(num_episodes=10)