from typing import Dict
import torch
import torch.nn as nn
from torch.profiler import record_function

from storm_kit.util_file import join_path, get_assets_path
from storm_kit.differentiable_robot_model.spatial_vector_algebra import quaternion_to_matrix, matrix_to_quaternion
from storm_kit.mpc.model import URDFKinematicModel

class ArmRollout(nn.Module):
    def __init__(self, cfg, task, value_function=None, viz_rollouts:bool = False, device=torch.device('cpu')):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.value_function = value_function
        self.viz_rollouts = viz_rollouts
        self.num_instances = cfg['num_instances']
        self.device = device

        assets_path = get_assets_path()
        # initialize dynamics model:
        self.dynamics_model = URDFKinematicModel(
            join_path(assets_path, cfg['model']['urdf_path']),
            batch_size=cfg['batch_size'],
            horizon=cfg['horizon'],
            num_instances=self.num_instances,
            ee_link_name=cfg['model']['ee_link_name'],
            link_names=cfg['model']['link_names'],
            dt_traj_params=cfg['model']['dt_traj_params'],
            control_space=cfg['control_space'],
            device=self.device)

    def forward(self, start_state: torch.Tensor, act_seq:torch.Tensor)->Dict[str, torch.Tensor]:
        return self.rollout_fn(start_state, act_seq)

    def compute_value_predictions(self, state_dict: torch.Tensor, act_seq):
        value_preds = None
        if self.value_function is not None:
            obs = self.task.compute_observations(state_dict)
            input_dict = {
                'obs': obs,
                'states': state_dict
            }
            value_preds = self.value_function.forward(input_dict, act_seq) 

        return value_preds #, state_dict

    def rollout_fn(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Parameters
        ----------
        action_seq: torch.Tensor [num_particles, horizon, d_act]
        """
        with record_function("robot_model"):
            state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        
        with record_function("compute_cost"):
            cost_seq, _= self.task.compute_cost(state_dict, act_seq)

        with record_function("compute_termination"):
            state_dict['prev_action'] = start_state['prev_action']
            term_seq, term_cost = self.task.compute_termination(state_dict, act_seq)

        # print(cost.shape, term_cost.shape)

        if term_cost is not None:
            cost_seq += term_cost

        with record_function("value_fn_inference"):
            value_preds = self.compute_value_predictions(state_dict, act_seq)


        sim_trajs = dict(
            actions=act_seq,
            costs=cost_seq,
            terminations=term_seq,
            ee_pos_seq=state_dict['ee_pos_seq'],
            value_preds=value_preds,
            rollout_time=0.0
        )

        if self.viz_rollouts:
            self.task.visualize_rollouts(sim_trajs)
        
        return sim_trajs
    
    def update_params(self, param_dict):
        return self.task.update_params(param_dict)