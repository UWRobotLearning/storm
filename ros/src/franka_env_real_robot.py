#!/usr/bin/env python
import copy, os
import torch
from hydra import initialize, compose
import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState

from storm_ros.srv import ReachGoal, ReachGoalResponse
from storm_ros.msg import GoalMsg
from storm_kit.learning.experience_collector import ExperienceCollector
from storm_kit.learning.replay_buffers import RobotBuffer
from storm_kit.learning.policies import MPCPolicy
from storm_kit.util_file import get_data_path

class FrankaRealRobotEnv():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.episode_length = cfg['task']['env']['episodeLength']
        self.n_dofs = cfg.task.rollout.n_dofs
        self.device = device
    
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.joint_names = rospy.get_param('~robot_joint_names', ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        
        self.default_config = torch.tensor([0.0, -0.7853, 0.0, -2.3561, 0.0, 1.5707, 0.7853])
        self.reset_state = torch.cat((self.default_config, torch.zeros_like(self.default_config), torch.zeros_like(self.default_config))).unsqueeze(0)
        self.control_dt = self.cfg.task.rollout.control_dt

        #Initialize reset policy
        self.reset_mpc_cfg = copy.deepcopy(self.cfg)
        self.reset_mpc_cfg.task.rollout.cost.goal_pose.weight = [0., 0.]
        self.reset_mpc_cfg.task.rollout.cost.joint_l2.weight = 100.0
        self.reset_mpc_cfg.task.rollout.cost.manipulability.weight = 0.0
        self.reset_mpc_cfg.mpc.mppi.horizon=20
        self.reset_policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=self.reset_mpc_cfg.mpc, rollout_cls=ArmReacher, device=self.device)

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

        self.default_ee_goal = torch.zeros(7, device=self.device)
        self.default_ee_goal[0] = 0.5
        self.default_ee_goal[1] = 0.0
        self.default_ee_goal[2] = 0.5
        self.default_ee_goal[3] = 0.0
        self.default_ee_goal[4] = 0.707
        self.default_ee_goal[5] = 0.707
        self.default_ee_goal[6] = 0.0
        self.default_ee_goal = self.default_ee_goal.unsqueeze(0)
        self.ee_goal = self.default_ee_goal.clone()


        #ROS Initialization
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        # self.service = rospy.Service('mpc', ReachGoal, self.go_to_goal)

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


    def collect_episodes(self,
                         num_episodes: int,
                         data_folder: str = None):
        
        collect_data = False
        if data_folder is not None:
            collect_data = True
        
        print('Collecting episodes')        
        input('Press any key to initiate reset')
        self.reset()
        for ep_num in range(num_episodes):
            input('Press any key to start collecting episode = {}'.format(ep_num))
            episode_buffer = self.collect_episode(collect_data=collect_data)
            if data_folder is not None:
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)
                filepath = data_folder + '/episode_{}.p'.format(ep_num)
                print('Saving episode to {}'.format(filepath))
                episode_buffer.save(filepath)

            input('Episode {}/{} done. Press any key to initiate reset'.format(ep_num, num_episodes))
            self.reset()


    def collect_episode(self, collect_data=False):
        tstep = 0
        ee_goal = self.ee_goal.clone()
        
        episode_buffer=None
        if collect_data:
            episode_buffer = RobotBuffer(capacity=self.episode_length, n_dofs=self.n_dofs)

        print(ee_goal)
        self.policy.reset()
        self.policy.update_goal(ee_goal=ee_goal)
        for i in range(self.episode_length):
            #only do something if state has been received
            if self.state_sub_on:
                input_dict = {}
                input_dict['states'] = torch.cat(
                    (self.obs_dict['states'], 
                        torch.as_tensor([tstep]).unsqueeze(0)),
                        dim=-1)
                input_dict = copy.deepcopy(input_dict) #we deepcopy here to ensure state does not change in the background
                
                #get mpc command
                command = self.policy.get_action(
                    obs_dict=input_dict)

                #publish mpc command
                mpc_command = JointState()
                mpc_command.header = self.command_header
                mpc_command.name = self.joint_names
                mpc_command.header.stamp = rospy.Time.now()
                mpc_command.position = command['q_des'][0].cpu().numpy()
                mpc_command.velocity = command['qd_des'][0].cpu().numpy()
                mpc_command.effort =  command['qdd_des'][0].cpu().numpy()

                self.command_pub.publish(mpc_command)

                if collect_data:
                    episode_buffer.add(
                        q_pos=input_dict['states'][:, 0:self.n_dofs], 
                        q_vel=input_dict['states'][:, self.n_dofs:2*self.n_dofs], 
                        q_acc=input_dict['states'][:, 2*self.n_dofs:3*self.n_dofs], 
                        q_pos_cmd=command['q_des'], 
                        q_vel_cmd=command['qd_des'], 
                        q_acc_cmd=command['qdd_des'],
                        ee_goal=ee_goal)

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
        return episode_buffer

    def step(self):
        pass

    def reset(self):
        tstep = 0
        self.reset_policy.reset()

        self.reset_policy.update_goal(joint_goal = self.reset_state)

        goal_q_pos = self.reset_state[:, 0:self.n_dofs]

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

        print('Randomizing ee_goal')
        self.ee_goal[:,0] = self.default_ee_goal[:,0] + 0.2*torch.rand(1).item() - 0.1
        self.ee_goal[:,1] = self.default_ee_goal[:,1] + 0.2*torch.rand(1).item() - 0.1
        print(self.ee_goal)

        return None


    def check_goal_reached(self, curr_error, delta_error_list):
        reached = False
        reached = curr_error <= 1e-2
        if len(delta_error_list) >= 20:
            reached = np.average(delta_error_list[-20:]) <= 1e-4
        return reached
    
    def close(self):
        self.command_pub.unregister()
        self.state_sub.unregister()


if __name__ == "__main__":
    from datetime import datetime
    import random
    import string
    
    torch.manual_seed(0)
    np.random.seed(0)


    rospy.init_node("robot_experience_collector", anonymous=True, disable_signals=True)    

    torch.set_default_dtype(torch.float32)
    initialize(config_path="../../content/configs/gym", job_name="mpc")
    config = compose(config_name="config", overrides=["task=FrankaReacherRealRobot"])
    control_dt = config.task.rollout.control_dt
    n_dofs = config.task.rollout.n_dofs

    device = torch.device('cuda', 0)


    #STORM Initialization
    obs_dim = 1
    act_dim = 1
    policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=config.mpc, device=device)

    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    data_folder =  os.path.join(get_data_path(), f'{now_str}_{rand_str}')
    experience_collector = RobotExperienceCollector(config, policy=policy, device=device)
    try:
        experience_collector.collect_episodes(num_episodes=50, data_folder=data_folder)
    except KeyboardInterrupt:
        print('Exiting')
        pass
    experience_collector.close()