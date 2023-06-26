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
            policy = None,
            device=torch.device('cpu')):
        
        super().__init__(obs_dim, act_dim, config, device=device)
        self.env_control_space = self.cfg['env_control_space']
        self.tensor_args = {'device': self.device, 'dtype' : torch.float32}
        self.controller = self.init_controller()
        self.dt = self.cfg.control_dt
        self.policy = policy

        self.state_filter = JointStateFilter(
            filter_coeff=self.cfg.state_filter_coeff, 
            device=self.device, #self.tensor_args['device'],
            n_dofs=self.cfg.n_dofs,
            dt=self.cfg.control_dt)
        
        self.prev_qdd_des = None

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
        st = time.time()
        states = obs_dict['states']
        states = states.to(self.device)
        self.state_filter.predict_internal_state(self.prev_qdd_des)

        # if(self.state_filter.cmd_joint_state is None):
        #     state_dict['velocity'] *= 0.0
        
        planning_state = self.state_filter.filter_joint_state(states)

        curr_action_seq, value, info = self.controller.forward(
            planning_state, calc_val=False, shift_steps=1)
        
        qdd_des = curr_action_seq[:, 0]
        qd_des = planning_state[:, 7:14] + qdd_des * self.dt
        q_des = planning_state[:, :7] + qd_des * self.dt
        # command = {
        #     'q_des': q_des,
        #     'qd_des': qd_des,
        #     'qdd_des': qdd_des
        # }
        if self.env_control_space == 'pos':
            command = q_des
        elif self.env_control_space == 'vel':
            command = qd_des
        elif self.env_control_space == 'vel_2':
            command = qd_des
        elif self.env_control_space == 'acc':
            command = qdd_des
        elif self.env_control_space == 'joint_impedance':
            command = {
                'q_des': q_des,
                'qd_des': qd_des,
                'qdd_des': qdd_des
            }

        self.prev_qdd_des = qdd_des #.clone()
        print(time.time()-st)
        return command

    def init_controller(self):
        # robot_yml = join_path(get_gym_configs_path(), robot_file)
        # robot_yaml = os.path.abspath(self.cfg.model.robot_collision_params.collision_spheres)
        # world_yaml = os.path.abspath(self.cfg.world_collision_def)

        # with open(robot_yml) as file:
            # robot_params = yaml.load(file, Loader=yaml.FullLoader)

        # world_yml = join_path(get_gym_configs_path(), collision_file)
        # with open(world_yaml) as file:
        #     world_params = yaml.load(file, Loader=yaml.FullLoader)
        world_params = self.cfg.world

        # mpc_yml_file = join_path(mpc_configs_path(), task_file)

        # with open(mpc_yml_file) as file:
        #     exp_params = yaml.load(file, Loader=yaml.FullLoader)
        rollout_params = self.cfg.rollout
        with open_dict(rollout_params):
            self.cfg['rollout']['num_instances'] = self.cfg['mppi']['num_instances']
            self.cfg['rollout']['horizon'] = self.cfg['mppi']['horizon']
            self.cfg['rollout']['num_particles'] = self.cfg['mppi']['num_particles']

        rollout_fn = self.get_rollout_fn(
            cfg=self.cfg['rollout'], device=self.device, world_params=world_params)        
        mppi_params = self.cfg.mppi
        dynamics_model = rollout_fn.dynamics_model
        with open_dict(mppi_params):
            mppi_params.d_action = dynamics_model.d_action
            mppi_params.action_lows =  [-rollout_params.model.max_acc] * dynamics_model.d_action # * torch.ones(dynamics_model.d_action, **self.tensor_args)
            mppi_params.action_highs = [ rollout_params.model.max_acc] * dynamics_model.d_action # * torch.ones(dynamics_model.d_action, **self.tensor_args)
        
        # init_q = torch.tensor(self.cfg.model.init_state, **self.tensor_args)
        init_q = torch.tensor(rollout_params.model.init_state, device=self.device)
        init_action = torch.zeros((mppi_params.num_instances, mppi_params.horizon, dynamics_model.d_action), **self.tensor_args)
        init_action[:,:,:] += init_q
        if self.cfg.control_space == 'acc':
            init_mean = init_action * 0.0 # device=device)
        elif self.cfg.control_space == 'pos':
            init_mean = init_action

        controller = MPPI(
            **mppi_params, 
            init_mean=init_mean,
            rollout_fn=rollout_fn,
            tensor_args=self.tensor_args)
        
        return controller

    def get_rollout_fn(self, **kwargs):
        rollout_fn = ArmReacher(**kwargs)
        return rollout_fn

    def update_goal(self, goal):
        self.controller.rollout_fn.update_params(goal)

    def reset(self):
        self.state_filter.reset()