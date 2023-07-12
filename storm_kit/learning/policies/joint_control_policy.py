from omegaconf import open_dict
import torch
import time

from torch.profiler import profile, record_function, ProfilerActivity


from storm_kit.learning.policies import Policy
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.rollout.arm_reacher import ArmReacher
from storm_kit.learning.policies import GaussianPolicy


class JointControlPolicy(Policy):
    def __init__(
            self, 
            obs_dim, 
            act_dim, 
            config, 
            device=torch.device('cpu')):
        
        super().__init__(obs_dim, act_dim, config, device=device)
        # self.env_control_space = self.cfg['env_control_space']
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.device = device
        self.dt = self.cfg.control_dt
        # self.rollout = rollout
        self.policy = GaussianPolicy(
            obs_dim, act_dim, config, device 
        )

        self.state_filter = JointStateFilter(
            filter_coeff=self.cfg.state_filter_coeff, 
            device=self.device, #self.tensor_args['device'],
            n_dofs=self.cfg.n_dofs,
            dt=self.dt)
        
        self.prev_command = None

    def forward(self, obs_dict):
        states = obs_dict['states']
        self.state_filter.predict_internal_state(self.prev_qdd_des)

        # if self.state_filter.cmd_joint_state is None:
        #     state_dict['velocity'] *= 0.0
        
        planning_state = self.state_filter.filter_joint_state(states)
        
        # state_tensor = torch.cat((
        #     filt_state['position'],
        #     filt_state['velocity'],
        #     filt_state['acceleration'],
        #     states[:, -1].unsqueeze(1)
        # ), dim=-1)

        curr_action_seq, value, info = self.controller.forward(
            planning_state, calc_val=False, shift_steps=1)
        
        qdd_des = curr_action_seq[:, 0]
        qd_des = planning_state[:, 7:14] + qdd_des * self.dt
        q_des = planning_state[:, :7] + qd_des * self.dt
        
        command = {
            'q_des': q_des,
            'qd_des': qd_des,
            'qdd_des': qdd_des
        }

        self.prev_qdd_des = qdd_des #.clone()
        return command


    def get_action(self, obs_dict, deterministic=False):
        state_dict = obs_dict['states']
        states = torch.cat((state_dict['q_pos'], state_dict['q_vel'], state_dict['q_acc'], state_dict['tstep']), dim=-1)
        states = states.to(self.device)
        self.state_filter.predict_internal_state(self.prev_qdd_des)

        # if self.state_filter.cmd_joint_state is None:
        #     state_dict['velocity'] *= 0.0
        
        planning_state = self.state_filter.filter_joint_state(states)

        curr_action_seq, value, info = self.controller.forward(
            planning_state, calc_val=False, shift_steps=1)

        q_acc_des = curr_action_seq[:, 0]
        q_vel_des = planning_state[:, 7:14] + q_acc_des * self.dt
        q_pos_des = planning_state[:, :7] + q_vel_des * self.dt

        command = {
            'q_pos_des': q_pos_des,
            'q_vel_des': q_vel_des,
            'q_acc_des': q_acc_des
        }

        self.prev_qdd_des = q_acc_des.clone()
        return command


    def update_goal(self, ee_goal=None, joint_goal=None, object_goal=None):
        print(ee_goal)
        self.controller.rollout_fn.update_params(ee_goal, goal_state=joint_goal)
    

    def reset(self):
        self.prev_command = None
        self.state_filter.reset()
