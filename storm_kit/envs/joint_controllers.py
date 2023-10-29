from typing import Optional
import torch

class InverseDynamicsController():
    def __init__(self, p_gains:torch.tensor, d_gains:torch.tensor, device:torch.device=torch.device('cpu')):
        self.p_gains = p_gains
        self.d_gains = d_gains
        self.num_dofs = self.p_gains.shape[-1]
        self.device = device
    
    def get_command(self, inertia_matrix:torch.tensor, q_pos:torch.tensor, q_vel:torch.tensor, 
                    q_pos_des:torch.tensor, q_vel_des:torch.tensor, q_acc_des: Optional[torch.tensor]=None):
        
        #compute desired acceleration
        q_acc_des_feedback = self.p_gains[:, 0:self.num_dofs] * (q_pos_des - q_pos) +\
                        self.d_gains[:, 0:self.num_dofs] * (q_vel_des - q_vel)  

        if q_acc_des is not None:
            q_acc_des = q_acc_des + q_acc_des_feedback

        torques = torch.einsum('ijk,ik->ij', inertia_matrix, q_acc_des)
        
        return torques        


class JointStiffnessController():
    def __init__(self, p_gains:torch.tensor, d_gains:torch.tensor, device:torch.device=torch.device('cpu')):
        self.p_gains = p_gains
        self.d_gains = d_gains
        self.num_dofs = self.p_gains.shape[-1]
        self.device = device

    def get_command(self, q_pos:torch.tensor, q_vel:torch.tensor, q_pos_des:torch.tensor, q_vel_des:torch.tensor):

        #compute desired acceleration
        torques = self.p_gains * (q_pos_des - q_pos) + \
            self.d_gains * (q_vel_des - q_vel)  
        return torques
