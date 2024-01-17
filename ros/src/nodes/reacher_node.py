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
from storm_kit.tasks import ArmReacher
from storm_kit.mpc.model import URDFKinematicModel
from storm_kit.learning.policies import MPCPolicy, JointControlWrapper
from storm_kit.learning.learning_utils import dict_to_device
from storm_kit.util_file import get_root_path
from storm_kit.envs.panda_real_robot_env import PandaRealRobotEnv
from storm_kit.envs.task_map import task_map


ROOT_DIR = os.path.join(get_root_path(), 'robot_data')

def get_clean_dataset(dataset):
    clean_dataset = dict.fromkeys(dataset.keys())
    for key in dataset:
        clean_data_list = dataset[key]
        # clean_data_list = [data.cpu().numpy() if isinstance(data, torch.Tensor) else data 
        #                     for data in dataset[key]]
        clean_data_np_array = np.concatenate(clean_data_list, axis=0)
        clean_dataset[key] = clean_data_np_array
    return clean_dataset 


class MPCReacherNode():
    def __init__(self, cfg) -> None:
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.joint_names = rospy.get_param('~robot_joint_names', None)
        self.save_data = rospy.get_param('~save_data', False)
        self.load_pretrained_policy = rospy.get_param('~load_pretrained', False)
        
        initialize(config_path="../../../content/configs/gym", job_name="mpc")
        self.config = compose(config_name="config", overrides=["task=FrankaReacherRealRobot"])
        self.control_dt = self.config.task.joint_control.control_dt
        self.n_dofs = self.config.task.n_dofs
        self.device = self.config.rl_device
        
        #STORM Initialization
        task_details = task_map[cfg.task_name]
        task_cls = task_details['task_cls']    
        dynamics_model_cls = task_details['dynamics_model_cls']

        self.env = PandaRealRobotEnv(
            cfg.task, device=cfg.rl_device,
            headless=True, safe_mode=False
        )

        self.policy = MPCPolicy(
            obs_dim=1, act_dim=1, 
            config=self.config.mpc, task_cls=task_cls, 
            dynamics_model_cls=dynamics_model_cls, device=self.device)
        # self.policy = JointControlWrapper(
        #     config=self.config.task.joint_control, policy=self.policy, device=self.device)

        if self.save_data:
            date_time = datetime.now().strftime("%m_%d_%Y_%H_%M")
            filename = 'mpc_data_{0}.p'.format(date_time)
            self.filepath = os.path.join(ROOT_DIR, filename)
            if not os.path.exists(ROOT_DIR):
                os.makedirs(ROOT_DIR)
            self.dataset = defaultdict(list)

        #buffers for different messages
        self.mpc_command = JointState()
        self.mpc_command.name = self.joint_names
        # self.gripper_state = JointState()
        self.command_header = None
        self.gripper_state = {
            'q_pos': torch.zeros(1,2),
            'q_vel': torch.zeros(1,2),
            'q_acc': torch.zeros(1,2)}
        
        self.robot_state = {
            'q_pos': torch.zeros(1,7, device=self.device),
            'q_vel': torch.zeros(1,7, device=self.device),
            'q_acc': torch.zeros(1,7, device=self.device)}

        self.goal_dict = {}
        self.ee_goal_pos = None
        self.ee_goal_quat = None

        #ROS Initialization
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        # self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        self.ee_goal_sub = rospy.Subscriber(self.ee_goal_topic, PoseStamped, self.ee_goal_callback, queue_size=1)
        self.control_freq = float(1.0/self.control_dt)
        self.rate = rospy.Rate(self.control_freq)

        self.state_sub_on = False
        self.goal_sub_on = False
        self.tstep = 0
        self.tstep_tensor = torch.as_tensor([self.tstep], device=self.config.rl_device).unsqueeze(0)
        self.start_t = None
        self.first_iter = True

    # def robot_state_callback(self, msg):
    #     self.state_sub_on = True
    #     self.command_header = msg.header
    #     #save gripper state
    #     # self.gripper_state.header = msg.header
    #     # self.gripper_state.position = msg.position[0:2]
    #     # self.gripper_state.velocity = msg.velocity[0:2]
    #     # self.gripper_state.effort = msg.effort[0:2]
    #     # self.gripper_state['position'] = np.array(msg.position[0:2])
    #     # self.gripper_state['velocity'] = np.array(msg.velocity[0:2])

    #     # #save robot state
    #     # self.robot_state.header = msg.header
    #     # self.robot_state.position = msg.position[2:]
    #     # self.robot_state.velocity = msg.velocity[2:]
    #     # self.robot_state.effort = msg.effort[2:]
    #     self.robot_state['q_pos'] = torch.tensor([msg.position], device=self.device)
    #     self.robot_state['q_vel'] = torch.tensor([msg.velocity], device=self.device)
    #     # self.robot_state['q_acc'] = torch.zeros_like(self.robot_state['q_vel'])

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
            if self.first_iter:
                state = self.env.get_state_dict()
            if self.goal_sub_on:
                #check if goal was updated
                if self.new_ee_goal:
                    param_dict = {'goal_dict': dict_to_device(self.goal_dict, device=self.device)}
                    self.policy.update_task_params(param_dict)
                    self.new_ee_goal = False
                # for k in self.robot_state.keys():
                #     self.policy_input['states'][k] = self.robot_state[k].clone().to(self.device)
                self.tstep_tensor[0,0] = self.tstep
                self.robot_state['tstep'] = self.tstep_tensor

                policy_input = {
                    'states': state
                }
                # print([v.device for k,v in policy_input['states'].items()])
                # self.policy_input['obs'] = self.robot_state_tensor.to(self.device)
                # for k in self.policy_input['states'].keys():
                #     print('in node', k, self.policy_input['states'][k].device)

                #get mpc command
                st=time.time()
                action, policy_info = self.policy.get_action(policy_input, deterministic=True)
                print(time.time()-st)
                next_obs, reward, done, info = self.env.step(
                    action, compute_cost=False, compute_termination=False)
                next_state = info['state']
                state = copy.deepcopy(next_state) 

                # #publish mpc command
                # self.mpc_command.header = self.command_header
                # self.mpc_command.header.stamp = rospy.Time.now()
                # self.mpc_command.position = command_tensor[0][0:7].cpu().numpy()
                # self.mpc_command.velocity = command_tensor[0][7:14].cpu().numpy()
                # self.mpc_command.effort =  command_tensor[0][14:21].cpu().numpy()

                # self.mpc_command.position = command['q_des'][0].cpu().numpy()
                # self.mpc_command.velocity = command['qd_des'][0].cpu().numpy()
                # self.mpc_command.effort =  command['qdd_des'][0].cpu().numpy()
                self.command_pub.publish(self.mpc_command)

                # if self.save_data:
                #     self.dataset['q_pos'].append(self.robot_state['q_pos'].cpu().numpy()) 
                #     self.dataset['q_vel'].append(self.robot_state['q_vel'].cpu().numpy())
                #     self.dataset['ee_goal'].append(self.goal_dict['ee_goal'].cpu().numpy())
                #     self.dataset['q_pos_cmd'].append(command_tensor[:, 0:7].cpu().numpy())
                #     self.dataset['q_vel_cmd'].append(command_tensor[:, 7:14].cpu().numpy())
                #     self.dataset['q_acc_cmd'].append(command_tensor[:, 14:21].cpu().numpy())
                    
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
            self.rate.sleep()
    
    def close(self):
        if self.save_data:
            print('Saving data to {0}'.format(self.filepath))
            clean_dataset = get_clean_dataset(self.dataset)
            
            with open(self.filepath, 'wb') as file:
                pickle.dump(clean_dataset, file)
        
        self.command_pub.unregister()
        # self.state_sub.unregister()
        self.ee_goal_sub.unregister()

@hydra.main(config_name="config", config_path="../../../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)    
    mpc_node = MPCReacherNode(cfg)
    try:
        mpc_node.control_loop()
    except KeyboardInterrupt:
        print('Exiting')
        mpc_node.close()

if __name__ == "__main__":
    main()