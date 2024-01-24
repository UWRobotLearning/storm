import torch 
import time 

class GymEnvWrapper():    
    def __init__(self, env, task=None):
        self.env = env
        self.task = task
        if self.task is not None:
            self.dummy_obs = torch.zeros([1,task.obs_dim], device=self.task.device)

    
    def step(self, action, compute_cost:bool=True, compute_termination:bool=True):
        if self.task is None:
            #mujoco
            if torch.is_tensor(action): action = action.cpu().numpy()
            return self.env.step(action)
        
        # state_dict = self.env.get_state_dict()
        next_state_dict, done_env = self.env.step(action)
        done = done_env
        cost_terms = None
        cost = 0.0
        obs = self.dummy_obs
        with torch.no_grad():
            if compute_cost or compute_termination:
                full_state_dict = self.task.compute_full_state(next_state_dict)
                obs = self.task.compute_observations(full_state_dict, compute_full_state=False, cost_terms=cost_terms)
            if compute_cost:
                cost, cost_terms = self.task.compute_cost(full_state_dict)
                cost = cost.item()
            done_task = False
            if compute_termination:
                done_task, term_cost, term_info = self.task.compute_termination(full_state_dict)
                done = done_env or done_task.item()
                cost += term_cost.item()

        return obs, cost, done, {'state': next_state_dict}

    def reset(self, rng=None):
        if self.task is None:
            return self.env.reset()
        with torch.no_grad():
            reset_data = self.task.reset(rng=rng)
            state_dict = self.env.reset(reset_data)
            obs = self.task.compute_observations(state_dict, compute_full_state=True)
        info = {'reset_data': reset_data, 'state': state_dict, 'done': False}
        return obs, info
    
    def seed(self, seed_val: None):
        pass