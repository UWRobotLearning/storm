#!/usr/bin/env python
import copy, os
import torch
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig
import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState

from storm_kit.learning.policies import MPCPolicy, JointControlWrapper
from storm_kit.tasks import ArmReacher

class FrankaRealRobotEnv():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.max_episode_length = cfg['env']['episodeLength']
        self.n_dofs = cfg.n_dofs
        self.device = device
        self.num_envs = cfg.env.get('num_envs', 1)
        self.robot_default_dof_pos = self.cfg["env"]["robot_default_dof_pos"]

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.joint_names = rospy.get_param('~robot_joint_names', ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        
        self.control_dt = self.cfg.joint_control.control_dt
        self.robot_default_dof_pos = torch.tensor(self.robot_default_dof_pos, device=self.device).unsqueeze(0)

        self.mpc_command = JointState()
        self.mpc_command.name = self.joint_names

        self.command_header = None
        self.gripper_state = {
            'q_pos': torch.zeros(1,2),
            'q_vel': torch.zeros(1,2),
            'q_acc': torch.zeros(1,2)}
        self.robot_state = {
            'q_pos': torch.zeros(1,7),
            'q_vel': torch.zeros(1,7),
            'q_acc': torch.zeros(1,7)}
        self.robot_state_tensor = None

        #ROS Initialization
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)

        self.control_freq = float(1.0/self.control_dt)
        self.rate = rospy.Rate(self.control_freq)

        self.state_sub_on = False
        self.tstep = 0
        self.start_t = None
        self.first_iter = True

        self.allocate_buffers()
        self.init_reset_policy()


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
        self.robot_state['q_pos'] = torch.tensor(msg.position).unsqueeze(0)
        self.robot_state['q_vel'] = torch.tensor(msg.velocity).unsqueeze(0)
        self.robot_state['q_acc'] = torch.zeros_like(self.robot_state['q_vel'])

        self.robot_state_tensor = torch.cat((
            self.robot_state['q_pos'],
            self.robot_state['q_vel'],
            self.robot_state['q_acc']
        )).unsqueeze(0)


    def allocate_buffers(self):
        # allocate buffers
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        # self.prev_action_buff = torch.zeros(
        #     (self.num_envs, self.n_dofs), device=self.device)
        self.target_buf = torch.zeros(
            (self.num_envs, 7), device=self.device
        )

    def _create_envs(self):
        pass

    def init_data(self):
        pass

    def pre_physics_steps(self, actions:torch.Tensor):
        pass

    def step(self, actions:torch.Tensor):
        #only do something if state has been received
        if self.state_sub_on:
            #publish mpc 
            self.mpc_command.header = self.command_header
            self.mpc_command.header.stamp = rospy.Time.now()
            self.mpc_command.position = actions[0][0:7].cpu().numpy()
            self.mpc_command.velocity = actions[0][7:14].cpu().numpy()
            self.mpc_command.effort =  actions[0][14:21].cpu().numpy()
            self.command_pub.publish(self.mpc_command)
            
            #update tstep
            if self.tstep == 0:
                rospy.loginfo('[FrankaRobotEnv]: Env Setup')
                self.start_t = rospy.get_time()
            self.tstep = rospy.get_time() - self.start_t

        else:
            if (not self.state_sub_on) and (self.first_iter):
                rospy.loginfo('[FrankaRobotEnv]: Waiting for robot state.')
        
        self.first_iter = False
        self.rate.sleep()
        state_dict = self.post_physics_step() 

        return state_dict, self.reset_buf
    

    def post_physics_step(self):
        self.progress_buf += 1                
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset()
        state_dict = self.get_state_dict()
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        return state_dict


    def get_state_dict(self):
        state_dict = copy.deepcopy(self.robot_state)
        state_dict['tstep'] = torch.as_tensor([self.tstep]).unsqueeze(0)
        return state_dict

    def reset(self, reset_data=None):
        input('Press enter to begin reset')
        print('[FrankaRobotEnv]: Resetting to default joint config')
        reset_data = {}
        reset_data['goal_dict'] = dict(joint_goal = self.robot_default_dof_pos)
        self.reset_policy.reset(reset_data)
        max_steps = 500
        goal_reached = False
        curr_q_pos = self.robot_state['q_pos']
        q_pos_goal = self.robot_default_dof_pos.cpu()
        curr_error = torch.norm(curr_q_pos - q_pos_goal, p=2)
        curr_num_steps = 0
        tstep, start_t = 0, 0

        while True:
            policy_input = {
                'states': self.get_state_dict()}

            command_tensor, policy_info = self.reset_policy.get_action(policy_input, deterministic=True)

            self.mpc_command.header = self.command_header
            self.mpc_command.header.stamp = rospy.Time.now()
            self.mpc_command.position = command_tensor[0][0:7].cpu().numpy()
            self.mpc_command.velocity = command_tensor[0][7:14].cpu().numpy()
            self.mpc_command.effort =  command_tensor[0][14:21].cpu().numpy()
            self.command_pub.publish(self.mpc_command)
        
            #update tstep
            if tstep == 0:
                start_t = rospy.get_time()
            tstep = rospy.get_time() - start_t

            curr_q_pos = self.robot_state['q_pos'].clone()
            curr_error = torch.norm(curr_q_pos - q_pos_goal, p=2).item()
            curr_num_steps += 1
            if (curr_error <= 0.005) or (curr_num_steps == max_steps -1):
                print('[FrankaRobotEnv]: Reset joint error = {}', curr_error)
                break
            self.rate.sleep()


        self.progress_buf[:] = 0
        self.reset_buf[:] = 0 
        input('Press enter to finish reset')
        return self.get_state_dict()
        # tstep = 0
        # self.reset_policy.reset()

        # self.reset_policy.update_goal(joint_goal = self.reset_state)

        # goal_q_pos = self.reset_state[:, 0:self.n_dofs]

        # goal_reached = False
        # curr_q_pos = self.obs_dict['states'][:, 0:self.n_dofs]
        # curr_error = torch.norm(curr_q_pos - goal_q_pos, p=2).item()
        # delta_error_list = []

        # while not goal_reached:
        #     #only do something if state has been received
        #     if self.state_sub_on:
        #         input_dict = {}
        #         input_dict['states'] = torch.cat(
        #             (self.obs_dict['states'],
        #                 torch.as_tensor([tstep]).unsqueeze(0)),
        #                 dim=-1)
                
        #         #get mpc command
        #         command = self.reset_policy.get_action(
        #             obs_dict=copy.deepcopy(input_dict))

        #         #publish mpc command
        #         mpc_command = JointState()
        #         mpc_command.header = self.command_header
        #         mpc_command.name = self.joint_names
        #         mpc_command.header.stamp = rospy.Time.now()
        #         mpc_command.position = command['q_des'][0].cpu().numpy()
        #         mpc_command.velocity = command['qd_des'][0].cpu().numpy()
        #         mpc_command.effort =  command['qdd_des'][0].cpu().numpy()

        #         self.command_pub.publish(mpc_command)

        #         #update tstep
        #         if tstep == 0:
        #             rospy.loginfo('[MPCPoseReacher]: Controller running')
        #             start_t = rospy.get_time()
        #         tstep = rospy.get_time() - start_t
                
        #         new_q_pos = self.obs_dict['states'][:, 0:self.n_dofs]
        #         new_error = torch.norm(new_q_pos - goal_q_pos, p=2).item()
        #         delta_error = abs(curr_error - new_error)
        #         delta_error_list.append(delta_error)
        #         goal_reached = self.check_goal_reached(new_error, delta_error_list)
        #         curr_error = copy.deepcopy(new_error)


        #     else:
        #         if (not self.state_sub_on) and (self.first_iter):
        #             rospy.loginfo('[MPCPoseReacher]: Waiting for robot state.')
            
        #     self.first_iter = False
        #     self.rate.sleep()
        
        # print('[Reset]: Goal Reached. curr_error={}, delta_error={}'.format(curr_error, delta_error))

        # print('Randomizing ee_goal')
        # self.ee_goal[:,0] = self.default_ee_goal[:,0] + 0.2*torch.rand(1).item() - 0.1
        # self.ee_goal[:,1] = self.default_ee_goal[:,1] + 0.2*torch.rand(1).item() - 0.1
        # print(self.ee_goal)

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

    def init_reset_policy(self):

        reset_cfg = compose(config_name="config", overrides=["task=FrankaReacherRealRobot", "mpc=FrankaReacherRealRobotMPC"])

        mpc_config = reset_cfg.mpc
        mpc_config.rollout.cost.goal_pose.weight = [0.0, 0.0]
        mpc_config.rollout.cost.joint_l2.weight = 5.0
        mpc_config.rollout.cost.ee_vel_twist.weight = 0.0
        mpc_config.rollout.cost.zero_q_vel.weight = 0.1
        mpc_config.rollout.cost.stop_cost.weight = 2.0

        # mpc_config.mppi.horizon = 10
        mpc_config.mppi.update_cov = False


        self.reset_policy = MPCPolicy(
            obs_dim=1, act_dim=1, 
            config=mpc_config, task_cls=ArmReacher, 
            device=self.device)
        self.reset_policy = JointControlWrapper(config=reset_cfg.task.joint_control, policy=self.reset_policy, device=self.device)


    # def collect_episodes(self,
    #                      num_episodes: int,
    #                      data_folder: str = None):
        
    #     collect_data = False
    #     if data_folder is not None:
    #         collect_data = True
        
    #     print('Collecting episodes')        
    #     input('Press any key to initiate reset')
    #     self.reset()
    #     for ep_num in range(num_episodes):
    #         input('Press any key to start collecting episode = {}'.format(ep_num))
    #         episode_buffer = self.collect_episode(collect_data=collect_data)
    #         if data_folder is not None:
    #             if not os.path.exists(data_folder):
    #                 os.makedirs(data_folder)
    #             filepath = data_folder + '/episode_{}.p'.format(ep_num)
    #             print('Saving episode to {}'.format(filepath))
    #             episode_buffer.save(filepath)

    #         input('Episode {}/{} done. Press any key to initiate reset'.format(ep_num, num_episodes))
    #         self.reset()


    # def collect_episode(self, collect_data=False):
    #     tstep = 0
    #     ee_goal = self.ee_goal.clone()
        
    #     episode_buffer=None
    #     if collect_data:
    #         episode_buffer = RobotBuffer(capacity=self.episode_length, n_dofs=self.n_dofs)

    #     print(ee_goal)
    #     self.policy.reset()
    #     self.policy.update_goal(ee_goal=ee_goal)
    #     for i in range(self.episode_length):
    #         #only do something if state has been received
    #         if self.state_sub_on:
    #             input_dict = {}
    #             input_dict['states'] = torch.cat(
    #                 (self.obs_dict['states'], 
    #                     torch.as_tensor([tstep]).unsqueeze(0)),
    #                     dim=-1)
    #             input_dict = copy.deepcopy(input_dict) #we deepcopy here to ensure state does not change in the background
                
    #             #get mpc command
    #             command = self.policy.get_action(
    #                 obs_dict=input_dict)

    #             #publish mpc command
    #             mpc_command = JointState()
    #             mpc_command.header = self.command_header
    #             mpc_command.name = self.joint_names
    #             mpc_command.header.stamp = rospy.Time.now()
    #             mpc_command.position = command['q_des'][0].cpu().numpy()
    #             mpc_command.velocity = command['qd_des'][0].cpu().numpy()
    #             mpc_command.effort =  command['qdd_des'][0].cpu().numpy()

    #             self.command_pub.publish(mpc_command)

    #             if collect_data:
    #                 episode_buffer.add(
    #                     q_pos=input_dict['states'][:, 0:self.n_dofs], 
    #                     q_vel=input_dict['states'][:, self.n_dofs:2*self.n_dofs], 
    #                     q_acc=input_dict['states'][:, 2*self.n_dofs:3*self.n_dofs], 
    #                     q_pos_cmd=command['q_des'], 
    #                     q_vel_cmd=command['qd_des'], 
    #                     q_acc_cmd=command['qdd_des'],
    #                     ee_goal=ee_goal)

    #             #update tstep
    #             if tstep == 0:
    #                 rospy.loginfo('[MPCPoseReacher]: Controller running')
    #                 start_t = rospy.get_time()
    #             tstep = rospy.get_time() - start_t
    #         else:
    #             if (not self.state_sub_on) and (self.first_iter):
    #                 rospy.loginfo('[MPCPoseReacher]: Waiting for robot state.')
            
    #         self.first_iter = False
    #         self.rate.sleep()
    #     return episode_buffer


@hydra.main(config_name="config", config_path="../../content/configs/gym")
def main(cfg: DictConfig):
    rospy.init_node("franka_real_robot_env", anonymous=True, disable_signals=True)    

    torch.set_default_dtype(torch.float32)
    # initialize(config_path="../../content/configs/gym", job_name="mpc")
    # config = compose(config_name="config", overrides=["task=FrankaReacherRealRobot"])
    # control_dt = config.task.rollout.control_dt
    # n_dofs = config.task.rollout.n_dofs

    device = torch.device('cuda', 0)


    # #STORM Initialization
    # obs_dim = 1
    # act_dim = 1
    # policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=config.mpc, device=device)

    # now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    # rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    # data_folder =  os.path.join(get_data_path(), f'{now_str}_{rand_str}')
    env = FrankaRealRobotEnv(cfg, device=device)
    env.reset()
    env.close()



if __name__ == "__main__":
    from datetime import datetime
    import random
    import string
    
    torch.manual_seed(0)
    np.random.seed(0)


    main()