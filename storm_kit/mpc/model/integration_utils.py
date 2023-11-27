#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch

def build_fd_matrix(horizon:int, device:torch.device=torch.device('cpu'), order:int=1, full_rank:bool=False):
    
    # if prev_state:
    #     # build order 1 fd matrix of horizon+order size
    #     fd1_mat = build_fd_matrix(horizon + order, device, order=1, diff_type=diff_type)
    #     # multiply order times to get fd_order matrix [h+order, h+order]
    #     fd_mat = fd1_mat
    #     for _ in range(order-1):
    #         fd_mat = fd_mat @ fd_mat
        # return [horizon,h+order]
        # fd_mat = fd_mat[:horizon, :]
        #fd_mat = torch.zeros((horizon, horizon + order),device=device, dtype=dtype)
        #one_t = torch.ones(horizon, device=device, dtype=dtype)
        #fd_mat[:horizon, :horizon] = torch.diag_embed(one_t)
        #print(torch.diag_embed(one_t, offset=1).shape, fd_mat.shape)
        #fd_mat += - torch.diag_embed(one_t, offset=1)[:-1,:]
    
    if full_rank:
        fd_mat = torch.eye(horizon, device=device)
        one_t = torch.ones(horizon//2, device=device)
        fd_mat[:horizon//2, :horizon//2] = torch.diag_embed(one_t)
        fd_mat[:horizon//2+1, :horizon//2+1] += - torch.diag_embed(one_t, offset=1)
        one_t = torch.ones(horizon//2, device=device)
        fd_mat[horizon//2:, horizon//2:] += - torch.diag_embed(one_t, offset=-1)
        fd_mat[horizon//2, horizon//2] = 0.0
        fd_mat[horizon//2, horizon//2-1] = -1.0
        fd_mat[horizon//2, horizon//2+1] = 1.0
    
    else:
        fd_mat = torch.zeros((horizon, horizon),device=device)
        one_t = torch.ones(horizon - 1, device=device)
        fd_mat[:horizon - 1, :horizon - 1] = -1.0 * torch.diag_embed(one_t)
        fd_mat += torch.diag_embed(one_t, offset=1)
        fd_mat = fd_mat[0:horizon-1, :]

    return fd_mat


def build_int_matrix(horizon, diagonal=0, device='cpu', dtype=torch.float32, order=1,
                     traj_dt=None):
    integrate_matrix = torch.tril(torch.ones((horizon, horizon), device=device, dtype=dtype), diagonal=diagonal)
    chain_list = [torch.eye(horizon, device=device, dtype=dtype)]
    if traj_dt is None:
        chain_list.extend([integrate_matrix for i in range(order)])
    else:
        diag_dt = torch.diag(traj_dt)
        
        for _ in range(order):
            chain_list.append(integrate_matrix)
            chain_list.append(diag_dt)
    integrate_matrix = torch.chain_matmul(*chain_list)
    return integrate_matrix


@torch.jit.script
def euler_integrate(
    q_0:torch.Tensor, 
    u:torch.Tensor, 
    diag_dt:torch.Tensor, 
    integrate_matrix:torch.Tensor) -> torch.Tensor:
    q_new = q_0 + torch.matmul(integrate_matrix, torch.matmul(diag_dt, u))
    return q_new

#@torch.jit.script
#TBD: fix
def tensor_step_jerk(state, act, state_seq, dt_h, n_dofs, integrate_matrix, fd_matrix=None):
    # type: (Tensor, Tensor, Tensor, Tensor, int, Tensor, Optional[Tensor]) -> Tensor
    # This is batch,n_dof
    q = state[:,:n_dofs]
    qd = state[:, n_dofs:2 * n_dofs]
    qdd = state[:, 2 * n_dofs:3 * n_dofs]

    diag_dt = torch.diag(dt_h)
    #qd_new = act
    # integrate velocities:
    qdd_new = qdd + torch.matmul(integrate_matrix, torch.matmul(diag_dt, act))
    qd_new = qd + torch.matmul(integrate_matrix, torch.matmul(diag_dt,qdd_new))
    q_new = q + torch.matmul(integrate_matrix, torch.matmul(diag_dt,qd_new))
    state_seq[:,:, :n_dofs] = q_new
    state_seq[:,:, n_dofs: n_dofs * 2] = qd_new
    state_seq[:,:, n_dofs * 2: n_dofs * 3] = qdd_new
    
    return state_seq

@torch.jit.script
def tensor_step_acc(
    state: torch.Tensor, 
    act: torch.Tensor, 
    state_seq: torch.Tensor, 
    dt_h: torch.Tensor, 
    integrate_matrix: torch.Tensor, 
    fd_matrix: torch.Tensor,
    n_dofs: int, 
    num_instances:int, 
    batch_size:int, 
    horizon:int) -> torch.Tensor:
    
    # This is batch,n_dof
    q = state[:, :, :n_dofs]
    qd = state[:, :, n_dofs:2*n_dofs]
    qdd_new = act.view(num_instances*batch_size, horizon, -1)
    diag_dt = torch.diag(dt_h)

    qd_new = torch.matmul(integrate_matrix, torch.matmul(diag_dt, qdd_new))
    qd_new = qd_new.view(num_instances, batch_size, horizon, -1)
    qd_new += qd.unsqueeze(1)

    q_new = torch.matmul(
        integrate_matrix, 
        torch.matmul(diag_dt, qd_new.view(num_instances*batch_size, horizon, -1)))
    q_new = q_new.view(num_instances, batch_size, horizon, -1)
    q_new += q.unsqueeze(1)

    # qddd_new = torch.matmul(diag_dt, torch.matmul(fd_matrix, qdd_new))
    # qddd_new = qddd_new.view(num_instances, batch_size, horizon, -1)
    # qddd_new = qddd_new.view(num_instances, batch_size, horizon, -1)

    qdd_new = qdd_new.view(num_instances, batch_size, horizon, -1)

    
    # qd_new = euler_integrate(qd, qdd_new, diag_dt, integrate_matrix)
    # q_new = euler_integrate(q, qd_new, diag_dt, integrate_matrix)

    state_seq[:, :, :, :n_dofs] = q_new
    state_seq[:, :, :, n_dofs:2*n_dofs] = qd_new
    state_seq[:, :, :, 2*n_dofs:3*n_dofs] = qdd_new
    # state_seq[:, :, :, 3*n_dofs:4*n_dofs] = qddd_new
    
    return state_seq

@torch.jit.script
def tensor_step_vel(
    state:torch.Tensor, 
    act:torch.Tensor, 
    state_seq:torch.Tensor, 
    dt_h:torch.Tensor, 
    integrate_matrix:torch.Tensor,
    fd_matrix:torch.Tensor,
    n_dofs:int, 
    num_instances:int,
    batch_size:int,
    horizon:int):
    
    # This is batch,n_dof
    q = state[:, :, 0:n_dofs]
    qd_new = act.view(num_instances*batch_size, horizon, -1)
    diag_dt = torch.diag(dt_h)

    q_new = torch.matmul(
        integrate_matrix, torch.matmul(diag_dt, qd_new))
    q_new = q_new.view(num_instances, batch_size, horizon, -1)
    q_new += q.unsqueeze(1)

    qdd_new = torch.matmul(diag_dt, torch.matmul(fd_matrix, qd_new))
    qdd_new = qdd_new.view(num_instances, batch_size, horizon, -1)
    qd_new = qd_new.view(num_instances, batch_size, horizon, -1)

    state_seq[:,:,:, :n_dofs] = q_new
    state_seq[:,:,:, n_dofs: 2*n_dofs] = qd_new
    state_seq[:,:,:, 2*n_dofs: 3*n_dofs] = qdd_new

    
    return state_seq

@torch.jit.script 
def tensor_step_pos(
        state:torch.Tensor, 
        act:torch.Tensor, 
        state_seq:torch.Tensor, 
        dt_h:torch.Tensor, 
        integrate_matrix:torch.Tensor,
        fd_matrix:torch.Tensor,
        n_dofs:int,
        num_instances:int,
        batch_size:int,
        horizon:int):
    
    
    # q = state[:, :, 0:n_dofs]
    
    #q_new = act #state[:,:n_dofs]
    q_new = act.view(num_instances*batch_size, horizon, -1)
    diag_dt = torch.diag(dt_h)

    qd_new = torch.matmul(diag_dt, torch.matmul(fd_matrix, q_new))
    qd_new = qd_new.view(num_instances, batch_size, horizon, -1)


    qdd_new = torch.matmul(diag_dt, torch.matmul(fd_matrix, qd_new))
    qdd_new = qdd_new.view(num_instances, batch_size, horizon, -1)

    # q = state[:, :n_dofs].unsqueeze(0).expand(state_seq.shape[0],-1,-1)
    # #print(q.shape, q_new.shape)
    # q = q_new #torch.cat((q, q_new), dim=1)

    
    #qd_new = act
    # integrate velocities:
    #q_new = q #q + torch.matmul(integrate_matrix_t0, torch.matmul(torch.diag(dt_h),qd_new))
    state_seq[:,:,:, :n_dofs] = q_new
    state_seq[:,:,:, n_dofs: 2*n_dofs] = qd_new
    state_seq[:,:,:, 2*n_dofs: 3*n_dofs] = qdd_new

    # state_seq[:,:, n_dofs: n_dofs * 2] = dt_diag @ fd_matrix @ q #qd_new
    # state_seq[:,:, n_dofs * 2: n_dofs * 3] = dt_diag @ dt_diag @ fd_matrix @ fd_matrix @ q

    #torch.matmul(torch.diag(dt_h), torch.matmul(fd_matrix, qd_new)

    
    return state_seq

def tensor_linspace(start_tensor, end_tensor, steps=10):
    #print(start_tensor.shape, end_tensor.shape)
    dist = end_tensor - start_tensor 
    interpolate_matrix = torch.ones((steps), device=start_tensor.device, dtype=start_tensor.dtype) / steps
    cum_matrix = torch.cumsum(interpolate_matrix, dim=0)
    #print(cum_matrix)
    interp_tensor = start_tensor + cum_matrix * dist
    return interp_tensor
    

                                                        
