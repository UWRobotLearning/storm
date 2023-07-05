#!/usr/bin/env python

#General imports
import copy
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

#STORM imports
# from storm_kit.mpc.task.reacher_task import ReacherTask
from storm_kit.learning.policies import MPCPolicy
from storm_kit.learning.replay_buffers import RobotBuffer
from storm_ros.srv import ReachGoal, ReachGoalResponse
from storm_ros.msg import JointMsg

class MPCReacherServer():
    def __init__(self) -> None:
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.joint_names = rospy.get_param('~robot_joint_names', ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        
        self.device = torch.device('cuda', 0)

        initialize(config_path="../../content/configs/gym", job_name="mpc")
        self.config = compose(config_name="config", overrides=["task=FrankaReacherRealRobot"])
        self.control_dt = self.config.task.rollout.control_dt
        self.n_dofs = self.config.task.rollout.n_dofs

        #STORM Initialization
        obs_dim = 1
        act_dim = 1
        self.policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=self.config.mpc, device=self.device)

        #Initialize reset policy
        self.reset_mpc_config = copy.deepcopy(self.config)
        self.reset_mpc_config.task.rollout.cost.goal_pose.weight = [0., 0.]
        self.reset_mpc_config.task.rollout.cost.joint_l2.weight = 100.0
        self.reset_mpc_config.task.rollout.cost.manipulability.weight = 0.0
        self.reset_mpc_config.mpc.mppi.horizon=20
        self.reset_policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=self.reset_mpc_config.mpc, device=self.device)

        #STORM Initialization
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
        self.service = rospy.Service('mpc', ReachGoal, self.go_to_goal)

        self.control_freq = float(1.0/self.control_dt)
        self.rate = rospy.Rate(self.control_freq)

        self.state_sub_on = False
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

    # def ee_goal_callback(self, msg):
    #     self.goal_sub_on = True
    #     self.new_ee_goal = True
    #     self.ee_goal_pos = torch.tensor([
    #         msg.pose.position.x,
    #         msg.pose.position.y,
    #         msg.pose.position.z])
    #     self.ee_goal_quat = torch.tensor([
    #         msg.pose.orientation.w,
    #         msg.pose.orientation.x,
    #         msg.pose.orientation.y,
    #         msg.pose.orientation.z])

    def go_to_goal(self, goal_req):
        self.tstep = 0
        self.policy.reset()
        self.reset_policy.reset()
        
        mode = goal_req.goal.MODE_FLAG
        if mode == 0:
            ee_goal = goal_req.goal.ee_goal
            ee_goal_pos = torch.tensor([
                ee_goal.position.x,
                ee_goal.position.y,
                ee_goal.position.z 
            ])
            ee_goal_quat = torch.tensor([
                ee_goal.orientation.w,
                ee_goal.orientation.x,
                ee_goal.orientation.y,
                ee_goal.orientation.z 
            ])
            goal = torch.cat((ee_goal_pos, ee_goal_quat), dim=-1).unsqueeze(0)
            self.go_to_ee_goal(goal)
            
        elif mode == 1:
            joint_goal = goal_req.goal.joint_goal

            q_pos_goal = torch.as_tensor(joint_goal.position)
            q_vel_goal = torch.as_tensor(joint_goal.velocity)

            goal = torch.cat((q_pos_goal, q_vel_goal), dim=-1).unsqueeze(0)
            self.go_to_joint_goal(goal)


        # for i in range(1000):
        #     #only do something if state has been received
        #     if self.state_sub_on:
        #         input_dict = {}
        #         input_dict['states'] = torch.cat(
        #             (self.obs_dict['states'],
        #                 torch.as_tensor([self.tstep]).unsqueeze(0)),
        #                 dim=-1                  
        #             )
        #         #get mpc command
        #         command = self.policy.get_action(obs_dict=copy.deepcopy(input_dict))

        #         #publish mpc command
        #         self.mpc_command.header = self.command_header
        #         self.mpc_command.header.stamp = rospy.Time.now()
        #         self.mpc_command.position = command['q_des'][0].cpu().numpy()
        #         self.mpc_command.velocity = command['qd_des'][0].cpu().numpy()
        #         self.mpc_command.effort =  command['qdd_des'][0].cpu().numpy()

        #         self.command_pub.publish(self.mpc_command)

        #         #update tstep
        #         if self.tstep == 0:
        #             rospy.loginfo('[MPCPoseReacher]: Controller running')
        #             self.start_t = rospy.get_time()
        #         self.tstep = rospy.get_time() - self.start_t

        #     else:
        #         if (not self.state_sub_on) and (self.first_iter):
        #             rospy.loginfo('[MPCPoseReacher]: Waiting for robot state.')
            
        #     self.first_iter = False
        #     self.rate.sleep()


        dummy = JointMsg()
        dummy.q_pos = 10
        dummy.q_vel = 10
        dummy.q_acc = 10
        dummy.q_pos_cmd = 10
        dummy.q_vel_cmd = 10
        dummy.q_acc_cmd = 10
        
        response = ReachGoalResponse()
        response.joint_data = [dummy]
        print(response)
        return response


    def go_to_ee_goal(self, ee_goal):
        tstep = 0
        self.policy.reset()
        self.policy.update_goal(ee_goal=ee_goal)
        for i in range(1000):
            #only do something if state has been received
            if self.state_sub_on:
                input_dict = {}
                input_dict['states'] = torch.cat(
                    (self.obs_dict['states'],
                        torch.as_tensor([tstep]).unsqueeze(0)),
                        dim=-1)
                
                #get mpc command
                command = self.policy.get_action(
                    obs_dict=copy.deepcopy(input_dict))

                #publish mpc command
                mpc_command = JointState()
                mpc_command.header = self.command_header
                mpc_command.name = self.joint_names
                mpc_command.header.stamp = rospy.Time.now()
                mpc_command.position = command['q_des'][0].cpu().numpy()
                mpc_command.velocity = command['qd_des'][0].cpu().numpy()
                mpc_command.effort =  command['qdd_des'][0].cpu().numpy()

                self.command_pub.publish(mpc_command)

                #update tstep
                if tstep == 0:
                    rospy.loginfo('[MPCPoseReacher]: Controller running')
                    start_t = rospy.get_time()
                tstep = rospy.get_time() - start_t
            else:
                if (not self.state_sub_on) and (self.first_iter):
                    rospy.loginfo('[MPCPoseReacher]: Waiting for robot state.')
            
            self.first_iter = False
            self.rate.sleep()


        dummy = JointMsg()
        dummy.q_pos = 10
        dummy.q_vel = 10
        dummy.q_acc = 10
        dummy.q_pos_cmd = 10
        dummy.q_vel_cmd = 10
        dummy.q_acc_cmd = 10
        
        response = ReachGoalResponse()
        response.joint_data = [dummy]
        return response  



    def go_to_joint_goal(self, joint_goal):
        tstep = 0
        self.reset_policy.reset()

        self.reset_policy.update_goal(joint_goal = joint_goal)

        goal_q_pos = joint_goal[:, 0:self.n_dofs]

        goal_reached = False
        curr_q_pos = self.obs_dict['states'][:, 0:self.n_dofs]
        curr_error = torch.norm(curr_q_pos - goal_q_pos, p=2).item()
        delta_error_list = []

        while not goal_reached:
            #only do something if state has been received
            if self.state_sub_on:
                input_dict = {}
                input_dict['states'] = torch.cat(
                    (self.obs_dict['states'],
                        torch.as_tensor([tstep]).unsqueeze(0)),
                        dim=-1)
                
                #get mpc command
                command = self.reset_policy.get_action(
                    obs_dict=copy.deepcopy(input_dict))

                #publish mpc command
                mpc_command = JointState()
                mpc_command.header = self.command_header
                mpc_command.name = self.joint_names
                mpc_command.header.stamp = rospy.Time.now()
                mpc_command.position = command['q_des'][0].cpu().numpy()
                mpc_command.velocity = command['qd_des'][0].cpu().numpy()
                mpc_command.effort =  command['qdd_des'][0].cpu().numpy()

                self.command_pub.publish(mpc_command)

                #update tstep
                if tstep == 0:
                    rospy.loginfo('[MPCPoseReacher]: Controller running')
                    start_t = rospy.get_time()
                tstep = rospy.get_time() - start_t
                
                new_q_pos = self.obs_dict['states'][:, 0:self.n_dofs]
                new_error = torch.norm(new_q_pos - goal_q_pos, p=2).item()
                delta_error = abs(curr_error - new_error)
                delta_error_list.append(delta_error)
                goal_reached = self.check_goal_reached(new_error, delta_error_list)
                curr_error = copy.deepcopy(new_error)


            else:
                if (not self.state_sub_on) and (self.first_iter):
                    rospy.loginfo('[MPCPoseReacher]: Waiting for robot state.')
            
            self.first_iter = False
            self.rate.sleep()
        
        print('[Reset]: Goal Reached. curr_error={}, delta_error={}'.format(curr_error, delta_error))


        dummy = JointMsg()
        dummy.q_pos = 10
        dummy.q_vel = 10
        dummy.q_acc = 10
        dummy.q_pos_cmd = 10
        dummy.q_vel_cmd = 10
        dummy.q_acc_cmd = 10
        
        response = ReachGoalResponse()
        response.joint_data = [dummy]
        return response  

    def check_goal_reached(self, curr_error, delta_error_list):
        # error = torch.norm(q_pos - q_pos_goal, p=2)
        # return  error.item() <= 1e-2
        reached = False
        reached = curr_error <= 1e-2
        if len(delta_error_list) >= 20:
            reached = np.average(delta_error_list[-20:]) <= 1e-4
        return reached
    
    def close(self):
        self.command_pub.unregister()
        self.state_sub.unregister()


if __name__ == "__main__":
    rospy.init_node("mpc_server", anonymous=True, disable_signals=True)    
    torch.set_default_dtype(torch.float32)

    mpc_node = MPCReacherServer()
    rospy.spin()
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print('Exiting')
    #     mpc_node.close()