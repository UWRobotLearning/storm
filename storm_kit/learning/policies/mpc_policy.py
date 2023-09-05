from omegaconf import open_dict
import torch
import time

from torch.profiler import profile, record_function, ProfilerActivity


from storm_kit.learning.policies import Policy
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.rollout.arm_reacher import ArmReacher
from storm_kit.mpc.utils.state_filter import JointStateFilter



class MPCPolicy(Policy):
    def __init__(
            self, 
            obs_dim, 
            act_dim, 
            config, 
            rollout_cls,
            sampling_policy = None,
            value_function = None,
            device=torch.device('cpu')):
        
        super().__init__(obs_dim, act_dim, config, device=device)
        self.tensor_args = {'device': self.device, 'dtype' : torch.float32}
        self.n_dofs = self.cfg.rollout.n_dofs

        self.sampling_policy = sampling_policy
        self.value_function = value_function
        self.rollout = self.init_rollout(rollout_cls) 
        self.controller = self.init_controller()
        self.dt = self.cfg.rollout.control_dt

        self.state_filter = JointStateFilter(
            filter_coeff=self.cfg.state_filter_coeff, 
            device=self.device, #self.tensor_args['device'],
            n_dofs=self.n_dofs,
            dt=self.dt)
        
        self.prev_qdd_des = None

    def forward(self, obs_dict):
        states = obs_dict['states']
        # self.state_filter.predict_internal_state(self.prev_qdd_des)

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
        
        q_acc_des = curr_action_seq[:, 0]
        command = self.state_filter.predict_internal_state(q_acc_des)
        # q_vel_des = planning_state[:, self.n_dofs:2*self.n_dofs] + qdd_des * self.dt
        # q_pos_des = planning_state[:, :self.n_dofs] + qd_des * self.dt
    
        command = {
            'q_pos_des': command['q_pos'],
            'q_vel_des': command['q_vel'],
            'q_acc_des': command['q_acc'],
        }

        self.prev_qdd_des = q_acc_des.clone() #.clone()
        return command


    def get_action(self, obs_dict, deterministic=False, num_samples=1):
        state_dict = obs_dict['states']
        # states = torch.cat(
        #     (state_dict['q_pos'], 
        #      state_dict['q_vel'], 
        #      state_dict['q_acc'], 
        #      state_dict['tstep']), dim=-1)
        # states = states.to(self.device)
        # self.state_filter.predict_internal_state(self.prev_qdd_des)
        # if self.state_filter.cmd_joint_state is None:
        #     state_dict['velocity'] *= 0.0
        planning_states = self.state_filter.filter_joint_state(state_dict)
        # planning_state_dict = state_dict
        # planning_state_dict['q_pos'] = planning_state[:, 0:self.n_dofs]
        # planning_state_dict['q_vel'] = planning_state[:, 0:self.n_dofs]
        # planning_state_dict['q_acc'] = planning_state[:, 0:self.n_dofs]
        
        curr_action_seq, value, info = self.controller.forward(
            planning_states, calc_val=False, shift_steps=1)


        q_acc_des = curr_action_seq[:, 0]
        command = self.state_filter.predict_internal_state(q_acc_des)


        # q_vel_des = planning_states['q_vel'] + q_acc_des * self.dt
        # q_pos_des = planning_states['q_pos'] + q_vel_des
        # q_vel_des = planning_state[:, self.n_dofs:2*self.n_dofs] + q_acc_des * self.dt
        # q_pos_des = planning_state[:, :self.n_dofs] + q_vel_des * self.dt

        command = {
            'q_pos_des': command['q_pos'],
            'q_vel_des': command['q_vel'],
            'q_acc_des': command['q_acc'],
        }

        self.prev_qdd_des = q_acc_des.clone()
        return command

    def init_controller(self):
        # world_params = self.cfg.world
        rollout_params = self.cfg.rollout

        # with open_dict(rollout_params):
        #     self.cfg['rollout']['num_instances'] = self.cfg['mppi']['num_instances']
        #     self.cfg['rollout']['horizon'] = self.cfg['mppi']['horizon']
        #     self.cfg['rollout']['num_particles'] = self.cfg['mppi']['num_particles']

        # rollout_fn = self.get_rollout_fn(
        #     cfg=self.cfg['rollout'], device=self.device, world_params=world_params)        
        mppi_params = self.cfg.mppi
        # dynamics_model = rollout_fn.dynamics_model
        with open_dict(mppi_params):
            mppi_params.d_action = self.n_dofs #dynamics_model.d_action
            mppi_params.action_lows =  [-1.0 * rollout_params.model.max_acc] * self.n_dofs #dynamics_model.d_action # * torch.ones(#dynamics_model.d_action, **self.tensor_args)
            mppi_params.action_highs = [ rollout_params.model.max_acc] * self.n_dofs #dynamics_model.d_action # * torch.ones(#dynamics_model.d_action, **self.tensor_args)
        
        # init_q = torch.tensor(self.cfg.model.init_state, **self.tensor_args)
        #TODO: This should be read from the environment
        init_q = torch.tensor(rollout_params.model.init_state, device=self.device)
        init_action = torch.zeros((mppi_params.num_instances, mppi_params.horizon, self.n_dofs), device=self.device)#dynamics_model.d_action), **self.tensor_args)
        if rollout_params.control_space == 'acc':
            init_mean = init_action * 0.0 
        elif rollout_params.control_space == 'pos':
            init_action[:,:,:] += init_q
            init_mean = init_action

        controller = MPPI(
            **mppi_params, 
            init_mean=init_mean,
            rollout_fn=self.rollout,
            tensor_args=self.tensor_args)
        
        return controller


    def init_rollout(self, rollout_cls):
        world_params = self.cfg.world
        rollout_params = self.cfg.rollout
        with open_dict(rollout_params):
           rollout_params['num_instances'] = self.cfg['mppi']['num_instances']
           rollout_params['horizon'] = self.cfg['mppi']['horizon']
           rollout_params['batch_size'] = self.cfg['mppi']['num_particles']

        return rollout_cls(
            cfg = rollout_params, world_params = world_params, value_function=self.value_function, viz_rollouts=self.cfg.viz_rollouts, device=self.device)

    def update_goal(self, goal_dict):
        self.controller.rollout_fn.update_params(goal_dict=goal_dict)
    
    def reset(self, reset_data):
        if 'goal_dict' in reset_data:
            self.update_goal(reset_data['goal_dict'])
        self.prev_qdd_des = None
        self.state_filter.reset()
        self.controller.reset_distribution()