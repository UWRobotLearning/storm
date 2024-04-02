#!/usr/bin/env python
import copy
import hydra
from datetime import datetime
from omegaconf import DictConfig
import os
from collections import defaultdict
import pickle
import numpy as np
import time
from hydra import compose, initialize
import isaacgym #this is annoying and needed because we are importing isaacgymenvs task map for using hydra.
import torch
# torch.multiprocessing.set_start_method('spawn',force=True)
# torch.set_num_threads(8)
# torch.backends.cudnn.benchmark = False
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

#ROS Imports
import rospy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
import rospkg
import roslaunch
import rospkg

#STORM imports
from storm_kit.learning.policies import MPCPolicy
from storm_kit.learning.learning_utils import dict_to_device
from storm_kit.util_file import get_root_path
from storm_kit.envs.panda_real_robot_env import PandaRealRobotEnv
from task_map import task_map

class MPCNode():
    def __init__(self, config):
        self.config = config
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.joint_names = rospy.get_param('~robot_joint_names', None)
        self.save_data = rospy.get_param('~save_data', False)
        self.load_pretrained_policy = rospy.get_param('~load_pretrained', False)
        # print("config used: ", self.config)
        self.control_dt = self.config.task.joint_control.control_dt
        self.n_dofs = self.config.task.joint_control.n_dofs
        self.device = self.config.rl_device
        
        #STORM Initialization
        task_details = task_map[config.task_name]
        task_cls = task_details['task_cls']    
        dynamics_model_cls = task_details['dynamics_model_cls']

        self.env = PandaRealRobotEnv(
            config.task, device=config.rl_device,
            headless=False, safe_mode=False,
            auto_reset_on_episode_end=False
        )

        self.policy = MPCPolicy(
            obs_dim=1, act_dim=1, 
            config=self.config.mpc, task_cls=task_cls, 
            dynamics_model_cls=dynamics_model_cls, device=self.config.rl_device)

        #buffers for different messages
        self.mpc_command = JointState()
        self.mpc_command.name = self.joint_names
        # self.gripper_state = JointState()
        self.command_header = None

        self.goal_dict = {}
        self.ee_goal_pos = None
        self.ee_goal_quat = None

        #ROS Initialization
        self.ee_goal_sub = rospy.Subscriber(self.ee_goal_topic, PoseStamped, self.ee_goal_callback, queue_size=1)
        self.control_freq = float(1.0/self.control_dt)
        self.rate = rospy.Rate(self.control_freq)

        self.state_sub_on = False
        self.goal_sub_on = False
        self.tstep = 0
        self.tstep_tensor = torch.as_tensor([self.tstep], device=self.config.rl_device).unsqueeze(0)
        self.start_t = None
        self.first_iter = True
        self.state = self.env.reset()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('storm_ros')

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(
            uuid, [os.path.join(pkg_path, "launch/interactive_marker_goal_pub.launch")])
        launch.start()


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

        goal = torch.cat((self.ee_goal_pos, self.ee_goal_quat), dim=-1).unsqueeze(0)
        self.goal_dict['ee_goal'] = goal

    def control_loop(self):
        while not rospy.is_shutdown():
            #only do something if state and goal have been received
            # if self.state_sub_on and self.goal_sub_on:
            if self.goal_sub_on:
                #check if goal was updated
                if self.new_ee_goal:
                    param_dict = {'goal_dict': dict_to_device(self.goal_dict, device=self.device)}
                    self.policy.update_task_params(param_dict)
                    self.env.update_goal(param_dict['goal_dict'])
                    self.new_ee_goal = False

                policy_input = {
                    'states': self.state
                }
                #get mpc command
                # st=time.time()

                action, policy_info = self.policy.get_action(policy_input, deterministic=True)
                # print(time.time()-st)
                self.state, _ = self.env.step(action)
                # next_state = info['state']
                # self.state = copy.deepcopy(next_state) 

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
            # self.rate.sleep()
    
    def close(self):        
        self.env.close()
        self.ee_goal_sub.unregister()

@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)    

    mpc_node = MPCNode(cfg)
    try:
        mpc_node.control_loop()
    except KeyboardInterrupt:
        print('Exiting')
        mpc_node.close()

if __name__ == "__main__":
    main()