import torch
import torch.nn as nn
from torch.profiler import record_function
from typing import Optional, Dict, List
from storm_kit.mpc.model.integration_utils import build_int_matrix, build_fd_matrix, tensor_step_acc, tensor_step_vel, tensor_step_pos, tensor_step_jerk

class SimplePushingModel(nn.Module):
    def __init__(
            self,
            n_dofs: int = 3,
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

        self.n_dofs = n_dofs
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

        #TODO: This hardcoding will be removed
        self.object_mass = 0.005
        self.robot_mass = 1.0
        self.object_radius = 0.03
        self.robot_radius = 0.05 #0.01

        # self.robot_mass_data = {
        #     'mass': 0.0,
        #     'inv_mass': 0.0
        # }

        # self.obj_mass_data = {
        #     'mass': self.object_mass,
        #     'inv_mass': 1.0 / self.object_mass
        # }

        self.M = torch.tensor([self.robot_mass] * self.n_dofs + [self.object_mass] * self.n_dofs, device=self.device)
        self.inv_M = 1.0 / self.M
        self.initial_step = True
        self.prev_state_buffer = {}

    def mass_matrix(self, full=False):
        if full:
            return torch.diag(self.M)
        return self.M


    def inverse_mass_matrix(self, full=False):
        if full:
            return torch.diag(self.inv_M)
        return self.inv_M

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
        if self.initial_step:
            for k in start_state_dict.keys():
                self.prev_state_buffer[k] = torch.zeros((self.num_instances, 10, start_state_dict[k].shape[-1]), device=self.device)
        #     # self.prev_state_buffer = torch.zeros((self.num_instances, 10, self.d_state), device=self.device, dtype=self.dtype)
                self.prev_state_buffer[k][:,:,:] = start_state_dict[k].unsqueeze(1)
                self.initial_step = False
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
        

        ee_state_seq = ee_state_seq.view((self.num_instances, curr_batch_size, num_traj_points, self.d_state))                
        ee_pos = ee_state_seq[:,:,:, 0:self.n_dofs]
        ee_vel = ee_state_seq[:,:,:, self.n_dofs:2*self.n_dofs]
        ee_acc = ee_state_seq[:,:,:, 2*self.n_dofs:3*self.n_dofs]
        tstep = ee_state_seq[:,:,:,-1]

        object_state_seq[:,:,0, 0:self.n_dofs] = start_state_dict['object_pos'].unsqueeze(1)
        object_state_seq[:,:,0, self.n_dofs:2*self.n_dofs] = start_state_dict['object_vel'].unsqueeze(1)

        object_pos = object_state_seq[:,:,:, 0:self.n_dofs]
        object_vel = object_state_seq[:,:,:, self.n_dofs:2*self.n_dofs]

        #run for loop to generate object states
        for t in range(0, self.horizon-1):

            coll_data = self.compute_sphere_collision(ee_pos[:, :, t], self.robot_radius, object_pos[:,:,t], self.object_radius)

            in_coll = coll_data['collision_count']
            normal = coll_data['normal']

            v_rel = object_vel[:,:,t] - ee_vel[:,:,t]
            v_rel_normal = torch.sum(v_rel * normal, dim=-1)


            tangent = v_rel -  normal * v_rel_normal.unsqueeze(-1)
            tangent_norm = torch.norm(tangent, dim=-1).unsqueeze(-1)
            tangent = torch.where(tangent_norm > 0, torch.div(tangent, tangent_norm), tangent)
            v_rel_tangent =  torch.sum(v_rel * tangent, dim=-1) 
            
            #normal impulse
            e = 0.0 #1.0 #1.0 #min(robot.material.restitution, obj.material.restitution)
            normal_impulse_magn = - (1 + e) * v_rel_normal
            normal_impulse_magn /= (self.robot_mass_data['inv_mass'] + self.obj_mass_data['inv_mass'])
            normal_impulse = normal_impulse_magn.unsqueeze(-1) * normal
            
            #frictional impulse
            #note: could also take averae of the two mu's (actually that might be more interprettable)
            # mu_static = np.sqrt(bodyA.material.mu_static ** 2 + bodyB.material.mu_static ** 2)            
            # mu_dynamic = np.sqrt(bodyA.material.mu_dynamic ** 2 + bodyB.material.mu_dynamic ** 2)            
            mu_static = 0.05 #0.005
            mu_dynamic = 0.05 #0.005

            friction_impulse_magn = - v_rel_tangent
            friction_impulse_magn /= (self.robot_mass_data['inv_mass'] + self.obj_mass_data['inv_mass'])

            coloumb_condition = torch.abs(friction_impulse_magn) < torch.abs(mu_static * normal_impulse_magn) 

            frictional_impulse = torch.where(
                coloumb_condition.unsqueeze(-1), 
                friction_impulse_magn.unsqueeze(-1) * tangent, 
                -1.0 * mu_dynamic * normal_impulse_magn.unsqueeze(-1) * tangent)
             
            # #apply to body
            # robot.velocity -= normal_impulse * robot.mass_data['inv_mass']
            obj_vel_update = (normal_impulse) * self.obj_mass_data['inv_mass'] #

            # self.robot_vel_buff[:,:,t,0:2] = self.robot_vel_buff[:,:,t, 0:2] * (1. - in_coll) + robot.velocity[:,:,0:2] * in_coll
            object_vel[:,:,t+1,0:3] = object_vel[:,:,t,0:3]  + obj_vel_update * in_coll    #obj.velocity[:,:,0:2] * in_coll
            
            # #calculate ground frictional impulse
            # tangent = object_vel[:,:,t+1,0:2]
            # tangent_norm = torch.norm(tangent, dim=-1).unsqueeze(-1)
            # tangent = torch.where(tangent_norm > 0, torch.div(tangent, tangent_norm), tangent)
            # v_rel_tangent = torch.sum(object_vel[:,:,t+1,0:2] * tangent, dim=-1)
            
            # mu_static = 1.0
            # mu_dynamic = 0.05

            # friction_impulse_magn = - v_rel_tangent
            # friction_impulse_magn /= self.obj_mass_data['inv_mass']

            # coloumb_condition = torch.abs(friction_impulse_magn) < mu_static * self.object_mass * 9.81
            # ground_frictional_impulse = torch.where(
            #     coloumb_condition.unsqueeze(-1), 
            #     friction_impulse_magn.unsqueeze(-1) * tangent, 
            #     -1.0 * mu_dynamic * self.object_mass * 9.81 * tangent)
             
            # # # #apply to body
            # ground_obj_vel_update = ground_frictional_impulse * self.obj_mass_data['inv_mass']
            # object_vel[:,:,t+1,0:2] += ground_obj_vel_update

            object_pos[:,:,t+1, 0:3] = object_pos[:,:,t] + object_vel[:,:,t+1] * self._dt_h[t]
            # # self.object_vel_buff[:,:, t, 0] = self.object_vel_buff[:,:,t-1, 0] * (1. - in_coll) + self.robot_vel_buff[:,:,t, 0] * in_coll
            # # self.object_vel_buff[:,:, t, 1] = self.object_vel_buff[:,:,t-1, 1] * (1. - in_coll) + self.robot_vel_buff[:,:,t, 1] * in_coll
            # robot_pos_update = self.robot_pos_buff[:,:,t-1, 0:2] + self.robot_vel_buff[:,:,t, 0:2] * self.dt 
            
            # self.robot_pos_buff[:,:,t, 0:2]  = self.robot_pos_buff[:,:,t, 0:2] * (1. - in_coll) + robot_pos_update * in_coll
            # self.object_pos_buff[:,:,t, 0:2] = self.object_pos_buff[:,:,t-1, 0:2] + self.object_vel_buff[:,:,t, 0:2] * self.dt 

        

        state_dict = {self.robot_keys[0]: ee_pos.to(inp_device),
                      self.robot_keys[1]: ee_vel.to(inp_device),
                      self.robot_keys[2]: ee_acc.to(inp_device),
                      'object_pos': object_pos.to(inp_device),
                      'object_vel': object_vel.to(inp_device),
                      'tstep': tstep.to(inp_device)}

        return state_dict


    def compute_sphere_collision(self, sphere1_center, sphere1_radius, sphere2_center, sphere2_radius):
        #find contact frame
        normal = sphere2_center - sphere1_center
        r = sphere1_radius + sphere2_radius
        dist = torch.linalg.vector_norm(normal, ord=2, dim=-1, keepdims=True)
        
        collision_count = torch.where(dist < r, 1.0, 0.0)
        penetration = torch.where(dist == 0.0, sphere1_radius, r - dist)
        
        I = torch.eye(normal.shape[-2], normal.shape[-1], device=self.device)
        collision_normal = torch.where(dist==0.0, I, normal/dist)
        
        #TODO?: Add contact point calculation

        return {
            'normal': collision_normal,
            'penetration': penetration,
            'collision_count': collision_count,
            'dist': dist
        }
   

    def reset(self):
        self.prev_state_buffer = {}
        self.initial_step = True

