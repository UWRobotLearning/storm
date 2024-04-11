import torch
import torch.nn as nn
import torch.nn.functional as F

# @torch.jit.script
class FrictionConeCost(nn.Module):
    # def __init__(self, m, d, J, mu, contact_points, gravitational_constant=9.81, device: torch.device = torch.device('cpu')):
    #     super().__init__()
    #     self.device = device
    #     self.m = m
    #     self.d = d.to(self.device)
    #     self.J = torch.tensor(J, device=self.device)
    #     self.g = gravitational_constant
    #     self.mu = mu
    #     self.debug = False
    #     self.friction_cost_weight = 1
    #     self.tangential_cost_weight = 0.0
    #     self.contact_points = contact_points.to(self.device)
    #     self.init_tensors()
        # self.G = self.compute_grasp_matrix(self.contact_points)
        # self.G_inverse = torch.pinverse(self.G)
    # def __init__(self, **kwargs):
    def __init__(self, object_params, weight:float, tangential_cost_weight:float=0.0, device:torch.device=torch.device('cpu')):
        G: torch.Tensor
        G_inverse: torch.Tensor

        super(FrictionConeCost, self).__init__()

        self.device = device #kwargs.get('device', torch.device('cpu'))
        self.m:float = object_params['m']
        self.d:torch.Tensor = torch.as_tensor(object_params['com'], device=self.device)
        self.J:torch.Tensor = torch.as_tensor(object_params['J'], device=self.device)
        self.mu:float = object_params['mu']
        self.g:float = object_params.get('gravitational_constant', 9.81)
        self.weight:float = weight
        self.tangential_cost_weight:float = tangential_cost_weight
        self.contact_points:torch.Tensor = torch.as_tensor(object_params['contact_points'], device=self.device)
        self.debug = False
        self.G:torch.Tensor = None
        self.G_inverse:torch.Tensor = None
        self.init_tensors()
        
    def init_tensors(self):
        self.G = self.compute_grasp_matrix(self.contact_points)
        self.G_inverse = torch.pinverse(self.G)

    @torch.jit.script
    def skew_symmetric(vec: torch.Tensor) -> torch.Tensor:
        sizes = list(vec.size()[:-1]) #obtains the size of the tensor up to the last dimension
        sizes.extend([3, 3]) #appends the new dimensions statically
        # zero = torch.zeros_like(vec[..., 0])
        mat = torch.zeros(sizes, dtype=vec.dtype, device=vec.device)
        mat[..., 0, 1] = -vec[..., 2]
        mat[..., 0, 2] = vec[..., 1]
        mat[..., 1, 0] = vec[..., 2]
        mat[..., 1, 2] = -vec[..., 0]
        mat[..., 2, 0] = -vec[..., 1]
        mat[..., 2, 1] = vec[..., 0]
        return mat
       
    def wGI(self, v_dot: torch.Tensor, omega: torch.Tensor, omega_dot: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        g_vector = torch.tensor([0, 0, -self.g], dtype=torch.float32, device=self.device).repeat(v_dot.shape[0], 1)
        term1_linear = self.m * (v_dot - torch.matmul(R, g_vector.unsqueeze(-1)).squeeze(-1))
        skew_omega = self.skew_symmetric(omega)
        skew_omega_dot = self.skew_symmetric(omega_dot)
        term2_angular = self.m * (torch.matmul(skew_omega_dot + torch.matmul(skew_omega, skew_omega), self.d.unsqueeze(-1)).squeeze(-1))
        term3_torque = torch.matmul(self.J, omega_dot.unsqueeze(-1)).squeeze(-1) + torch.matmul(torch.matmul(skew_omega, self.J), omega.unsqueeze(-1)).squeeze(-1)
        wGI = torch.cat((term1_linear + term2_angular, term3_torque), dim=-1)
        wGI = -wGI
        return wGI
    
    def compute_grasp_matrix(self, contact_points: torch.Tensor) -> torch.Tensor: #using non prehensile tray obj mpc paper
        n = contact_points.shape[0] 
        G = []
        for i in range(n):
            #R is identity and p is the position of contact points
            R = torch.eye(3, device=self.device)
            p = contact_points[i]
            p_hat = self.skew_symmetric(p)
            #compute adjoint transformation matrix
            Ad_T = torch.cat((R, torch.zeros(3, 3, device=self.device)), 1)
            Ad_T = torch.cat((Ad_T, torch.cat((p_hat, torch.zeros(3, 3, device=self.device)), 1)), 0)
            if self.debug:
                print("AD_T", Ad_T)
            #compute Bc,i
            Bc = torch.cat((torch.eye(3, device=self.device), torch.zeros(3, 3, device=self.device)), 0)
            G.append(Ad_T @ Bc)
        G_matrix = torch.hstack(G)
        return G_matrix

    # @torch.jit.script
    def forward(self, v_dot: torch.Tensor, omega: torch.Tensor, omega_dot: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        v_dot, omega, omega_dot, R = v_dot.to(self.device), omega.to(self.device), omega_dot.to(self.device), R.to(self.device)
        # total_cost = torch.zeros(1, batch_size, horizon, device=self.device)
        w_gi = self.wGI(v_dot, omega, omega_dot, R)
        G = self.G
        if self.debug:
            print("G shape", G.shape)  # 6*12
        Fc = torch.matmul(self.G_inverse, -w_gi.unsqueeze(-1)).squeeze(-1)
        if self.debug:
            Fc_total = Fc[..., :3] + Fc[..., 3:6] + Fc[..., 6:9] + Fc[..., 9:12]
            # print(Fc_total)
        tangential_indices = torch.tensor([0, 1, 3, 4, 6, 7, 9, 10], device=self.device)
        f_t = torch.index_select(Fc, dim=-1, index=tangential_indices)
        normal_indices = torch.tensor([2, 5, 8, 11], device=self.device)
        f_n = torch.index_select(Fc, dim=-1, index=normal_indices)
        f_t = f_t.reshape(*f_t.shape[:-1], 4, 2)
        l2_norm_ft = torch.norm(f_t, p=2, dim=-1)
        tangential_force_norm =  l2_norm_ft 
        friction_cone_violation_norm = l2_norm_ft - self.mu*torch.abs(f_n) #logcosh(l2_norm_ft - 0.7 * f_n)
        mask = friction_cone_violation_norm > 0
        friction_cost = friction_cone_violation_norm * mask
        total_cost = self.weight*friction_cost.sum(dim=-1) + self.tangential_cost_weight*tangential_force_norm.sum(dim=-1) #sum over all contact points
        return 1 * total_cost
