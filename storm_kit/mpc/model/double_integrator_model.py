import torch
import torch.nn as nn
from torch.profiler import record_function
from typing import Optional, Dict, List
from storm_kit.mpc.model.integration_utils import build_int_matrix, build_fd_matrix, tensor_step_acc, tensor_step_vel, tensor_step_pos, tensor_step_jerk
from storm_kit.differentiable_robot_model.coordinate_transform import CoordinateTransform

class DoubleIntegratorModel(nn.Module):
    def __init__(
            self,
            batch_size: int = 1000,
            horizon: int = 5,
            num_instances: int = 1,
            dt_traj_params: Optional[torch.Tensor] = None, 
            control_space: str = 'acc',
            robot_keys: List[str] = ['ee_pos', 'ee_vel', 'ee_acc'],
            device: torch.device = torch.device('cpu')):
        
        super().__init__()
        self.batch_size = batch_size
        self.horizon = horizon
        self.num_instances = num_instances
        self.dt_traj_params = dt_traj_params
        self.control_space = control_space
        self.num_traj_points = self.horizon
        self.robot_keys = robot_keys
        self.device = device

        self.n_dofs = 2
        self.d_state = 3 * self.n_dofs + 1

        self.ee_state_seq = torch.zeros(self.num_instances, self.batch_size, self.num_traj_points, self.d_state, device=self.device)
        self.object_state_seq = torch.zeros(self.num_instances, self.batch_size, self.num_traj_points, self.d_state, device=self.device)
        self._integrate_matrix = build_int_matrix(self.num_traj_points, device=self.device)
        self._fd_matrix = build_fd_matrix(self.num_traj_points, device=self.device, order=1)

        self.control_space = control_space        
        if control_space == 'pos':
            self.step_fn = tensor_step_pos
        elif control_space == 'vel':
            self.step_fn = tensor_step_vel
        elif control_space == 'acc':
            self.step_fn = tensor_step_acc
        elif control_space == 'jerk':
            self.step_fn = tensor_step_jerk
        
        if self.num_traj_points <= 1:
            dt_array = [self.dt_traj_params['base_dt']] * int(1.0 * self.num_traj_points) 
        else:
            dt_array = [dt_traj_params['base_dt']] * int(dt_traj_params['base_ratio'] * self.num_traj_points)
            smooth_blending = torch.linspace(dt_traj_params['base_dt'], dt_traj_params['max_dt'], steps=int((1 - dt_traj_params['base_ratio']) * self.num_traj_points)).tolist()
            dt_array += smooth_blending
            self.dt = dt_traj_params['base_dt']
        
        if len(dt_array) != self.num_traj_points:
            dt_array.insert(0,dt_array[0])
        
        self._dt_h = torch.tensor(dt_array, device=self.device)
        self._traj_tstep = torch.matmul(self._integrate_matrix, self._dt_h)

    def forward(self, state_dict: Dict[str, torch.Tensor], act:torch.Tensor, dt) -> Dict[str,torch.Tensor]: 
        return self.get_next_state(state_dict, act, dt)

    def get_next_state(self, state_dict: [str, torch.Tensor], act:torch.Tensor, dt)->Dict[str, torch.Tensor]:
        """ Does a single step from the current state
        Args:
        state_dict: current state
        act: action
        dt: time to integrate
        Returns:
        next_state
        """
        ee_pos = state_dict[self.robot_keys[0]]
        ee_vel = state_dict[self.robot_keys[1]]
        ee_acc = state_dict[self.robot_keys[2]]
        if self.control_space == 'pos':
            ee_vel = ee_vel * 0.0
            ee_acc = ee_acc * 0.0
            ee_pos = act
        elif self.control_space == 'vel':
            ee_acc = ee_acc * 0.0
            ee_vel = act
            ee_pos += ee_vel * dt
        elif self.control_space == 'acc':
            ee_acc = act
            ee_vel += ee_acc * dt
            ee_pos += ee_vel * dt
        elif self.control_space == 'jerk':
            ee_acc += act * dt
            ee_vel += ee_acc * dt
            ee_pos += ee_vel * dt

        next_state_dict = {
            self.robot_keys[0]: ee_pos,
            self.robot_keys[1]: ee_vel,
            self.robot_keys[2]: ee_acc
        }
        return next_state_dict
    
    def tensor_step(self, 
            state: torch.Tensor, act: torch.Tensor, 
            state_seq: torch.Tensor, batch_size:int, horizon:int) -> torch.Tensor:
        
        inp_device = act.device
        state = state.to(self.device)
        act = act.to(self.device)
        # nth_act_seq = self.integrate_action(act)
        state_seq = self.step_fn(state, act, state_seq, self._dt_h, self._integrate_matrix, self._fd_matrix, self.n_dofs, self.num_instances, batch_size, horizon)
        state_seq[:, :,:, -1] = self._traj_tstep # timestep array
        return state_seq.to(inp_device)
        
        
    @torch.jit.export
    def rollout_open_loop(self, start_state_dict: Dict[str, torch.Tensor], act_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        inp_device = act_seq.device
        act_seq = act_seq.to(self.device)
    
        # add start state to prev state buffer:
        # if self.initial_step is None:
        #     # self.prev_state_buffer = torch.zeros((self.num_instances, 10, self.d_state), device=self.device, dtype=self.dtype)
        #     self.prev_state_buffer[:,:,:] = start_state.unsqueeze(1)
        # self.prev_state_buffer = self.prev_state_buffer.roll(-1, dims=1)
        # self.prev_state_buffer[:,-1,:] = start_state

        ee_state_seq = self.ee_state_seq
        object_state_seq = self.object_state_seq
        curr_batch_size = self.batch_size
        num_traj_points = self.num_traj_points
        
        # curr_state = self.prev_state_buffer[:,-1:,:self.n_dofs * 3]

        #First we get the ee states neglecting collisions
        start_ee_pos = start_state_dict[self.robot_keys[0]]
        start_ee_vel = start_state_dict[self.robot_keys[1]]
        start_ee_acc = start_state_dict[self.robot_keys[2]]

        curr_ee_state = torch.cat((start_ee_pos, start_ee_vel, start_ee_acc), dim=-1)
        curr_ee_state = curr_ee_state.unsqueeze(1)
        with record_function("tensor_step"):
            ee_state_seq = self.tensor_step(curr_ee_state, act_seq, ee_state_seq, curr_batch_size, num_traj_points)
        
        ee_state_seq = ee_state_seq.view((self.num_instances, curr_batch_size, num_traj_points, 7))                
        ee_pos = ee_state_seq[:,:,:,0:2]
        ee_vel = ee_state_seq[:,:,:,2:4]
        ee_acc = ee_state_seq[:,:,:,4:6]
        tstep = ee_state_seq[:,:,:,-1]

        
        state_dict = {self.robot_keys[0]: ee_pos.to(inp_device),
                      self.robot_keys[1]: ee_vel.to(inp_device),
                      self.robot_keys[2]: ee_acc.to(inp_device),
                      'tstep': tstep.to(inp_device)}

        return state_dict



