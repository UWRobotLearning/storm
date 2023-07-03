#!/usr/bin/env python
import numpy as np
import rospy

from storm_ros.srv import ReachGoal, ReachGoalResponse
from storm_ros.msg import GoalMsg



def send_goal():
    print('Waiting for service')
    rospy.wait_for_service('mpc')
    print('Service running')

    try:
        mpc = rospy.ServiceProxy('mpc', ReachGoal)
        goal_msg = GoalMsg()
        goal_msg.MODE_FLAG = 0
        goal_msg.ee_goal.position.x = 0.30675235390663147 + 0.1
        goal_msg.ee_goal.position.y = 0.004366477020084858
        goal_msg.ee_goal.position.z = 0.48579496145248413
        goal_msg.ee_goal.orientation.x =  0.7018908262252808
        goal_msg.ee_goal.orientation.y =  0.7120890021324158
        goal_msg.ee_goal.orientation.z =  -0.016190776601433754
        goal_msg.ee_goal.orientation.w =  0.0040412466041743755
        goal_msg.joint_goal.name=['hello', 'hello2', 'hello3', 'hello4', 'hello5', 'hello6', 'hello7']
        goal_msg.joint_goal.position = [0.0] * 7
        goal_msg.joint_goal.velocity = [0.0] * 7
        goal_msg.joint_goal.effort = [0.0] * 7


        service_req = ReachGoal()
        service_req.goal = goal_msg
        
        ret = mpc(goal_msg)

        return ret
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)



if __name__ == "__main__":
    send_goal()
