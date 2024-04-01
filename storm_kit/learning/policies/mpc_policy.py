from typing import Dict
from omegaconf import open_dict
import torch
from torch.distributions import Normal, MultivariateNormal, TransformedDistribution
from torch.profiler import record_function
import time
from storm_kit.learning.policies import Policy
from storm_kit.mpc.control import MPPI
from torch.utils._pytree import tree_map
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
            vf = None,
            qf = None,
            device=torch.device('cpu')):
        
        super().__init__(obs_dim, act_dim, config, device=device)
        self.tensor_args = {'device': self.device, 'dtype' : torch.float32}
        # self.rollout = self.init_rollout(task_cls) 
        self.controller = self.init_controller(
            task_cls, dynamics_model_cls,
            sampling_policy, vf ,qf)
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

    @torch.no_grad()
    @torch.autocast(device_type='cuda', enabled=False, dtype=torch.float16)
    def get_action(self, obs_dict, deterministic=False): #, num_samples=1):
        state_dict = obs_dict['states']
        # state_dict_to_device = tree_map(lambda x: x.to('cuda' if torch.cuda.is_available() else 'cpu'), state_dict)
        for key, tensor in state_dict.items():
            pass
            # print(f"Device of tensor '{key}':", tensor.device)
        with record_function('mpc_policy:get_action'):
            # st = time.time()
            # curr_action_seq, _, _ = self.controller.sample(
            #     state_dict, shift_steps=1, deterministic=deterministic)#, calc_val=False, num_samples=num_samples)
            # torch.cuda.synchronize()
            # print("Time to get action: ", time.time() - st)
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record()
            curr_action_seq, _, _ = self.controller.sample(
                state_dict, shift_steps=1, deterministic=deterministic)
            torch.cuda.synchronize()
            en.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            print("Time to get action: ", st.elapsed_time(en))

        action = curr_action_seq[:, 0]
        info = {}
        return action, info
        
    # @torch.no_grad()
    # @torch.autocast(device_type='cuda', enabled=False, dtype=torch.float16)
    # def get_action(self, obs_dict, deterministic=False):
    #     state_dict = obs_dict['states']
    #     #wrap the core logic in a new function for compilation.
    #     def compiled_action_logic(state_dict, deterministic):
    #         with torch.profiler.record_function('mpc_policy:get_action'):
    #             st = torch.cuda.Event(enable_timing=True)
    #             en = torch.cuda.Event(enable_timing=True)
    #             st.record()
    #             curr_action_seq, _, _ = self.controller.sample(
    #                 state_dict, shift_steps=1, deterministic=deterministic)
    #             torch.cuda.synchronize()
    #             en.record()
    #             torch.cuda.synchronize()  # Wait for the events to be recorded!
    #             print("Time to get action: ", st.elapsed_time(en))
    #         action = curr_action_seq[:, 0]
    #         return action
        
    #     #use torch.compile to optimize the compiled_action_logic function.
    #     optimized_action_logic = torch.compile(compiled_action_logic)
        
    #     #ensure that tensors are moved to the appropriate device before calling the compiled function.
    #     state_dict_to_device = tree_map(lambda x: x.to('cuda' if torch.cuda.is_available() else 'cpu'), state_dict)
    #     action = optimized_action_logic(state_dict_to_device, deterministic)
    #     info = {}
    #     return action, info

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
                        sampling_policy, vf=None, qf=None):
        # world_params = self.cfg.world
        # rollout = self.init_rollout(task_cls)
        # task_params = self.cfg.rollout
        task, dynamics_model = self.init_rollout(task_cls, dynamics_model_cls)
        print('dynamics_model: ', dynamics_model)
        task_params = self.cfg.task
        model_params = self.cfg.model

        mppi_params = self.cfg.mppi
        with open_dict(mppi_params):
            mppi_params.d_action = dynamics_model.n_dofs #dynamics_model.d_action
            print("mppi_params.d_action: ", mppi_params.d_action)
            mppi_params.action_lows =  [-1.0 * task_params.max_acc] * dynamics_model.n_dofs #dynamics_model.d_action # * torch.ones(#dynamics_model.d_action, **self.tensor_args)
            mppi_params.action_highs = [ task_params.max_acc] * dynamics_model.n_dofs #dynamics_model.d_action # * torch.ones(#dynamics_model.d_action, **self.tensor_args)
        
        # init_q = torch.tensor(self.cfg.model.init_state, **self.tensor_args)
        #TODO: This should be read from the environment
        init_q = torch.tensor(model_params['init_state'], device=self.device)
        # self.init_action = torch.zeros((mppi_params.num_instances, mppi_params.horizon, task_params.n_dofs), device=self.device)#dynamics_model.d_action), **self.tensor_args)
        self.init_action = torch.zeros((mppi_params.horizon, dynamics_model.n_dofs), device=self.device)#dynamics_model.d_action), **self.tensor_args)
        # if task_params.control_space == 'acc':
        #     init_mean = self.init_action
        if model_params['control_space'] == 'pos':
            self.init_action[:,:,:] += init_q
        init_mean = self.init_action

        # controller = MPPI(
        #     **mppi_params, 
        #     init_mean=init_mean,
        #     task=task,
        #     dynamics_model=dynamics_model,
        #     sampling_policy=sampling_policy,
        #     vf=vf, qf=qf,
        #     tensor_args=self.tensor_args)
    # return controller

        controller = torch.compile(MPPI(
            **mppi_params, 
            init_mean=init_mean,
            task=task,
            dynamics_model=dynamics_model,
            sampling_policy=sampling_policy,
            vf=vf, qf=qf,
            tensor_args=self.tensor_args))
        print("controller: ", controller)
        return controller


    def init_rollout(self, task_cls, dynamics_model_cls):
        world_cfg = self.cfg.world
        rollout_cfg = self.cfg.task
        model_cfg = self.cfg.model
        print("model_cfg: ", model_cfg)
        
        with open_dict(rollout_cfg):
           rollout_cfg['horizon'] = self.cfg['mppi']['horizon']
           rollout_cfg['batch_size'] = self.cfg['mppi']['num_particles']
           rollout_cfg['dt_traj_params'] = model_cfg['dt_traj_params']

        with open_dict(model_cfg):
           model_cfg['horizon'] = self.cfg['mppi']['horizon']
           model_cfg['batch_size'] = self.cfg['mppi']['num_particles']

        task = task_cls(cfg = rollout_cfg, world_cfg = world_cfg, device=self.device)
        dynamics_model = torch.jit.script(dynamics_model_cls(
            cfg = model_cfg,
            # batch_size=task_params['batch_size'],
            # horizon=task_params['horizon'],
            # ee_link_name=task_params['model']['ee_link_name'],
            # link_names=task_params['model']['link_names'],
            # dt_traj_params=task_params['model']['dt_traj_params'],
            # control_space=task_params['control_space'],
            device=self.device
        ))
        # dynamics_model = dynamics_model_cls(
        #     cfg = model_cfg,
        #     # batch_size=task_params['batch_size'],
        #     # horizon=task_params['horizon'],
        #     # ee_link_name=task_params['model']['ee_link_name'],
        #     # link_names=task_params['model']['link_names'],
        #     # dt_traj_params=task_params['model']['dt_traj_params'],
        #     # control_space=task_params['control_space'],
        #     device=self.device
        # )

        return task, dynamics_model


    def update_task_params(self, param_dict):
        self.controller.task.update_params(param_dict)
        
    def reset(self, reset_data=None):
        self.controller.reset(reset_data)
    
    def set_prediction_metrics(self, value_metrics=None):
        self.controller.set_prediction_metrics(value_metrics)
    