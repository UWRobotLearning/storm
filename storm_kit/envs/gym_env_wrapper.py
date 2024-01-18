import torch 

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
        state_dict, done_env = self.env.step(action)
        done = done_env
        cost_terms = None
        cost = 0.0
        obs = self.dummy_obs
        if compute_cost or compute_termination:
            full_state_dict = self.task.compute_full_state(state_dict)
            obs = self.task.compute_observations(full_state_dict, compute_full_state=False, cost_terms=cost_terms)
        if compute_cost:
            cost, cost_terms = self.task.compute_cost(full_state_dict)
            cost = cost.item()
        done_task = False
        if compute_termination:
            done_task, term_cost, term_info = self.task.compute_termination(full_state_dict)
            done = done_env or done_task.item()
            cost += term_cost.item()

        # assert torch.allclose(state_dict['q_pos'], full_state_dict['q_pos_seq'])
        # assert torch.allclose(obs[:,0:7], state_dict['q_pos'])
        return obs, cost, done, {'state': state_dict}

    def reset(self, rng=None):
        if self.task is None:
            return self.env.reset()
        reset_data = self.task.reset(rng=rng)
        state_dict = self.env.reset(reset_data)
        obs = self.task.compute_observations(state_dict, compute_full_state=True)
        info = {'reset_data': reset_data, 'state': state_dict}
        return obs, info
    
    def seed(self, seed_val: None):
        pass