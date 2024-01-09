from typing import Dict
from omegaconf import open_dict
import torch
from torch.distributions import Normal, MultivariateNormal, TransformedDistribution
import time


from storm_kit.learning.policies import Policy
from storm_kit.mpc.control import MPPI
# from storm_kit.mpc.rollout.arm_rollout import ArmRollout
from storm_kit.util_file import join_path, get_assets_path

class MPCPolicy(Policy):
    def __init__(
            self, 
            obs_dim, 
            act_dim, 
            config, 
            task_cls,
            dynamics_model_cls,
            sampling_policy = None,
            value_function = None,
            device=torch.device('cpu')):
        
        super().__init__(obs_dim, act_dim, config, device=device)
        self.tensor_args = {'device': self.device, 'dtype' : torch.float32}
        self.n_dofs = self.cfg.rollout.n_dofs
        # self.rollout = self.init_rollout(task_cls) 
        self.controller = self.init_controller(
            task_cls, dynamics_model_cls,
            sampling_policy, value_function)
        self.prev_command_time = time.time()
        # self.dt = self.cfg.rollout.control_dt
        
    def forward(self, obs_dict, calc_val:bool=False): #calc_val:bool=False
        states = obs_dict['states']
        distrib_info, aux_info =  self.controller.optimize(
            states, shift_steps=1, calc_val=calc_val) #value #calc_val=calc_val, 
        
        mean = distrib_info['mean'][:, 0]
        scale_tril = distrib_info['scale_tril']
        dist = MultivariateNormal(loc=mean, scale_tril=scale_tril)
        value = distrib_info['optimal_value'] if 'optimal_value' in distrib_info else 0.
        aux_info['base_policy_value'] = distrib_info['base_policy_value'] if 'base_policy_value' in distrib_info else 0.
        
        return dist, value, aux_info

    def get_action(self, obs_dict, deterministic=False): #, num_samples=1):
        state_dict = obs_dict['states']
        curr_action_seq, _, _ = self.controller.sample(
            state_dict, shift_steps=1, deterministic=deterministic)#, calc_val=False, num_samples=num_samples)
        action = curr_action_seq[:, 0]
        return action

    def log_prob(self, input_dict: Dict[str, torch.Tensor], actions: torch.Tensor):
        dist = self.forward(input_dict)
        log_prob = dist.log_prob(actions)
        return log_prob
    
    def entropy(self, input_dict: Dict[str, torch.Tensor], num_samples:int = 1):
        dist = self.forward(input_dict)
        actions = dist.rsample(sample_shape=torch.Size([num_samples]))
        log_prob = dist.log_prob(actions)
        return actions, log_prob.mean(0)

    def init_controller(self, 
                        task_cls, dynamics_model_cls,
                        sampling_policy, value_function):
        # world_params = self.cfg.world
        # rollout = self.init_rollout(task_cls)
        # rollout_params = self.cfg.rollout
        task, dynamics_model = self.init_rollout(task_cls, dynamics_model_cls)
        rollout_params = self.cfg.rollout

        mppi_params = self.cfg.mppi
        with open_dict(mppi_params):
            mppi_params.d_action = self.n_dofs #dynamics_model.d_action
            mppi_params.action_lows =  [-1.0 * rollout_params.max_acc] * self.n_dofs #dynamics_model.d_action # * torch.ones(#dynamics_model.d_action, **self.tensor_args)
            mppi_params.action_highs = [ rollout_params.max_acc] * self.n_dofs #dynamics_model.d_action # * torch.ones(#dynamics_model.d_action, **self.tensor_args)
        
        # init_q = torch.tensor(self.cfg.model.init_state, **self.tensor_args)
        #TODO: This should be read from the environment
        init_q = torch.tensor(rollout_params.model.init_state, device=self.device)
        # self.init_action = torch.zeros((mppi_params.num_instances, mppi_params.horizon, self.n_dofs), device=self.device)#dynamics_model.d_action), **self.tensor_args)
        self.init_action = torch.zeros((mppi_params.horizon, self.n_dofs), device=self.device)#dynamics_model.d_action), **self.tensor_args)
        # if rollout_params.control_space == 'acc':
        #     init_mean = self.init_action
        if rollout_params.control_space == 'pos':
            self.init_action[:,:,:] += init_q
        init_mean = self.init_action

        controller = MPPI(
            **mppi_params, 
            init_mean=init_mean,
            # rollout_fn=rollout,
            task=task,
            dynamics_model=dynamics_model,
            sampling_policy=sampling_policy,
            value_function=value_function,
            tensor_args=self.tensor_args)
        return controller


    def init_rollout(self, task_cls, dynamics_model_cls):
        world_params = self.cfg.world
        rollout_params = self.cfg.rollout
        with open_dict(rollout_params):
        #    rollout_params['num_instances'] = self.cfg['mppi']['num_instances']
           rollout_params['horizon'] = self.cfg['mppi']['horizon']
           rollout_params['batch_size'] = self.cfg['mppi']['num_particles']
           rollout_params['dt_traj_params'] = rollout_params['model']['dt_traj_params']
                
        task = task_cls(cfg = rollout_params, world_params = world_params, device=self.device)
        dynamics_model = dynamics_model_cls(
            join_path(get_assets_path(), rollout_params['model']['urdf_path']),
            batch_size=rollout_params['batch_size'],
            horizon=rollout_params['horizon'],
            # num_instances=self.cfg.mppi['num_instances'],
            ee_link_name=rollout_params['model']['ee_link_name'],
            link_names=rollout_params['model']['link_names'],
            dt_traj_params=rollout_params['model']['dt_traj_params'],
            control_space=rollout_params['control_space'],
            device=self.device
        )

        return task, dynamics_model

        # return ArmRollout(cfg = rollout_params, task=task, value_function=self.value_function, viz_rollouts=self.cfg.viz_rollouts, device=self.device)

        # return rollout_cls(
        #     cfg = rollout_params, world_params = world_params, value_function=self.value_function, viz_rollouts=self.cfg.viz_rollouts, device=self.device)

    def update_task_params(self, param_dict):
        self.controller.task.update_params(param_dict)
        
    def reset(self, reset_data=None):
        self.controller.reset(reset_data)