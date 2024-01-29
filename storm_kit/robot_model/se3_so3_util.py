"""
SE3 SO3 utilities
====================================
@author: gsutanto
@comment: implemented from "A Mathematical Introduction to Robotic Manipulation"
          textbook by Murray et al., page 413-414
@comment: Mohak - made all functions operate on batches of matrices and 
          torch.jit compatible.
"""

from typing import Tuple
import torch


assert_epsilon = 1.0e-3


def integrateAxisAngle(axis_angle, omega, dt):
    R_curr = expMapso3(getSkewSymMatFromVec3(axis_angle))
    R_delta = expMapso3(getSkewSymMatFromVec3(omega * dt))
    R_next = torch.matmul(R_delta, R_curr)
    axis_angle_next = getVec3FromSkewSymMat(logMapSO3(R_next))
    return axis_angle_next
    R = T[:3, :3]

    angle = torch.norm(axis_angle)
    if angle > epsilon:
        axis = axis_angle / angle
        quat = axis_angle.new_zeros(4)
        quat[:3] = axis * torch.sin(angle / 2.0)
        quat[3] = torch.cos(angle / 2.0)
    else:
        quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=axis_angle.device, dtype=axis_angle.dtype
        )
    quat = quat / torch.norm(quat)
    return quat

@torch.jit.script
def convertQuaternionToAxisAngle(quat:torch.Tensor, alpha:float=0.05, epsilon:float=1.0e-15):
    # if not torch.is_tensor(quat):
    #     quat = torch.Tensor(quat)
    # assert quat.shape[0] == 4
    # assert (torch.norm(quat) > 1.0 - alpha) and (torch.norm(quat) < 1.0 + alpha)
    quat = quat / torch.norm(quat, dim=-1)
    angle = 2.0 * torch.acos(quat[:, 0])[:, None]
    axis = quat[:, 1:] / (torch.sin(angle / 2.0) + epsilon)
    axis_angle = axis * angle
    return axis_angle


def getSkewSymMatFromVec3(omega):
    omega = omega.reshape(3)
    omegahat = omega.new_zeros((omega.shape[0], 3, 3))
    sign_multiplier = -1
    for i in range(3):
        for j in range(i + 1, 3):
            omegahat[i, j] = sign_multiplier * omega[3 - i - j]
            omegahat[j, i] = -sign_multiplier * omega[3 - i - j]
            sign_multiplier = -sign_multiplier
    return omegahat

@torch.jit.export
def getVec3FromSkewSymMat(omegahat:torch.Tensor, epsilon:float=1.0e-14)->torch.Tensor:
    # assert torch.norm(torch.diag(omegahat)) < assert_epsilon, (
    #     "omegahat = \n%s" % omegahat
    # )
    # for i in range(3):
    #     for j in range(i + 1, 3):
    #         v1 = omegahat[i, j]
    #         v2 = omegahat[j, i]
    #         err = torch.abs(v1 + v2)
    #         assert err < epsilon, "err = %f >= %f = epsilon" % (err, epsilon)
    omega = omegahat.new_zeros(omegahat.shape[0], 3)
    omega[:,0] = 0.5 * (omegahat[:, 2, 1] - omegahat[:, 1, 2])
    omega[:,1] = 0.5 * (omegahat[:, 0, 2] - omegahat[:, 2, 0])
    omega[:,2] = 0.5 * (omegahat[:, 1, 0] - omegahat[:, 0, 1])
    return omega


def getKseehatFromWrench(wrench):
    assert wrench.shape[0] == 6
    v = wrench[:3]
    omega = wrench[3:6]
    omegahat = getSkewSymMatFromVec3(omega)
    kseehat = wrench.new_zeros((4, 4))
    kseehat[:3, :3] = omegahat
    kseehat[:3, 3] = v
    return kseehat


def getWrenchFromKseehat(kseehat, epsilon=1.0e-14):
    assert torch.norm(kseehat[3, :]) < assert_epsilon, "kseehat = \n%s" % kseehat
    v = kseehat[:3, 3].reshape((3, 1))
    omegahat = kseehat[:3, :3]
    omega = getVec3FromSkewSymMat(omegahat, epsilon).reshape((3, 1))
    wrench = torch.stack([v, omega])
    assert wrench.shape[0] == 6, "wrench.shape[0] = %d" % wrench.shape[0]
    return wrench.reshape((6,))


def getHomogeneousTransformMatrixFromAxes(orig, axis_x, axis_y, axis_z):
    T = torch.eye(4)
    T[:3, 0] = axis_x
    T[:3, 1] = axis_y
    T[:3, 2] = axis_z
    T[:3, 3] = orig
    return T


def getAxesFromHomogeneousTransformMatrix(T):
    assert torch.norm(T[3, :3]) < assert_epsilon
    assert torch.abs(T[3, 3] - 1.0) < assert_epsilon

    axis_x = T[:3, 0]
    axis_y = T[:3, 1]
    axis_z = T[:3, 2]
    orig = T[:3, 3]

    return orig, axis_x, axis_y, axis_z


def getInverseHomogeneousTransformMatrix(T, epsilon=1.0e-14):
    assert torch.norm(T[3, :3]) < assert_epsilon
    assert torch.abs(T[3, 3] - 1.0) < assert_epsilon
    R = T[:3, :3]
    assert (
        torch.abs(torch.abs(torch.det(R)) - 1.0) < assert_epsilon
    ), "det(R) = %f" % torch.det(R)
    p = T[:3, 3]
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype)
    Rinv = R.T
    pinv = -torch.matmul(Rinv, p)
    Tinv[:3, :3] = Rinv
    Tinv[:3, 3] = pinv
    return Tinv

@torch.jit.script
def logMapSO3(rot:torch.Tensor, epsilon:float=1.0e-14) -> torch.Tensor:
    # assert R.shape[0] == 3
    # assert R.shape[1] == 3
    # assert (
    #     torch.abs(torch.abs(torch.det(R)) - 1.0) < assert_epsilon
    # ), "det(R) = %f" % torch.det(R)

    # half_traceR_minus_one = (torch.trace(R) - 1.0) / 2.0
    # if half_traceR_minus_one < -R.new_ones(1):
    #     print("Warning: half_traceR_minus_one = %f < -1.0" % half_traceR_minus_one)
    #     half_traceR_minus_one = -R.new_ones(1)
    # if half_traceR_minus_one > 1.0:
    #     print("Warning: half_traceR_minus_one = %f > 1.0" % half_traceR_minus_one)
    #     half_traceR_minus_one = R.new_ones(1)
    half_traceR_minus_one = 0.5 * (rot.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1.0) #/ 2.0
    theta = torch.acos(torch.clamp(half_traceR_minus_one, -1.0, 1.0))[:, None, None]
    # omegahat = (R - R.T) / ((2.0 * torch.sin(theta)) + epsilon)
    omegahat = (rot - rot.transpose(-1, -2)) / ((2.0 * torch.sin(theta)) + epsilon)
    return theta * omegahat

@torch.jit.script
def expMapso3(omegahat:torch.Tensor, epsilon:float=1.0e-14):
    # assert omegahat.shape[0] == 3
    # assert omegahat.shape[1] == 3
    omega = getVec3FromSkewSymMat(omegahat) #, epsilon)

    norm_omega = torch.norm(omega)
    exp_omegahat = (
        torch.eye(3, device=omegahat.device, dtype=omegahat.dtype)
        + ((torch.sin(norm_omega) / (norm_omega + epsilon)) * omegahat)
        + (
            ((1.0 - torch.cos(norm_omega)) / (norm_omega + epsilon) ** 2)
            * torch.matmul(omegahat, omegahat)
        )
    )
    return exp_omegahat

@torch.jit.script
def logMapSE3(rot:torch.Tensor, trans:torch.Tensor, epsilon:float=1.0e-14) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    omegahat = logMapSO3(rot, epsilon)
    omega = getVec3FromSkewSymMat(omegahat, epsilon)
    norm_omega = torch.norm(omega, dim=-1)[:, None, None]

    omegahat_times_omegahat = torch.einsum('bij, bjk -> bik', omegahat, omegahat)
    I = torch.eye(3, device=trans.device, dtype=trans.dtype)

    #computing Ainv
    term1 = I - 0.5 * omegahat
    term2_num = (2.0 * torch.sin(norm_omega)) - (norm_omega * (1.0 + torch.cos(norm_omega)))
    term2_den =  ((2 * (norm_omega**2) * torch.sin(norm_omega)) + epsilon)
    term2  = torch.divide(term2_num, term2_den) * omegahat_times_omegahat
    Ainv = term1 + term2
    v = torch.matmul(Ainv, trans.unsqueeze(-1))
    return omegahat, omega, v.squeeze(-1)


def expMapse3(kseehat , epsilon=1.0e-14):
    # assert kseehat.shape[0] == 4
    # assert kseehat.shape[1] == 4
    # assert torch.norm(kseehat[3, :]) < assert_epsilon
    omegahat = kseehat[:, :3, :3]
    exp_omegahat = expMapso3(omegahat, epsilon)

    omega = getVec3FromSkewSymMat(omegahat, epsilon)
    norm_omega = torch.norm(omega, dim=-1)

    A = (
        torch.eye(3, device=kseehat.device, dtype=kseehat.dtype)
        + (((1.0 - torch.cos(norm_omega)) / (norm_omega + epsilon) ** 2) * omegahat)
        + (
            ((norm_omega - torch.sin(norm_omega)) / ((norm_omega + epsilon) ** 3))
            * torch.matmul(omegahat, omegahat)
        )
    )
    v = kseehat[:3, 3]
    exp_kseehat = torch.eye(4, device=kseehat.device, dtype=kseehat.dtype)
    exp_kseehat[:3, :3] = exp_omegahat
    exp_kseehat[:3, 3] = torch.matmul(A, v)
    return exp_kseehat



if __name__ == "__main__":
    import time
    import torch
    from se3_so3_util_old import logMapSE3 as logMapSE3Old
    from spatial_vector_algebra import x_rot, y_rot, z_rot

    #some simple profiling tests
    #generate random SE(3) transforms
    batch_size = 10000
    device = torch.device('cuda:0')
    trans = torch.randn(batch_size, 3, device=device)
    #random rotations
    xrot = x_rot(torch.ones(batch_size, device=device) * (0.5 * torch.pi))
    yrot = y_rot(torch.ones(batch_size, device=device) * torch.pi)
    zrot = z_rot(torch.ones(batch_size, device=device) * torch.pi)

    rot = xrot #* yrot * zrot

    #check logmap from new and old
    old_rot = []
    old_trans = []
    st = time.time()
    for i in range(trans.shape[0]):
        T = torch.zeros(4,4, device=device)
        T[-1, -1] = 1.0
        T[0:3, 0:3] = rot[i]
        T[0:3, -1] = trans[i]
        old_i = logMapSE3Old(T)
        old_rot.append(old_i[:3,:3].unsqueeze(0))
        old_trans.append(old_i[:3,-1].unsqueeze(0))

    old_rot = torch.cat(old_rot, dim=0)
    old_trans = torch.cat(old_trans, dim=0)
    old_time = time.time()-st

    #do it bunch of times for jit
    for _ in range(5):
        st = time.time()
        new_rot, _, new_trans = logMapSE3(rot, trans)
        new_time = time.time()-st
    assert torch.allclose(old_rot, new_rot)
    assert torch.allclose(old_trans, new_trans)
    print('Passed, Old time = {}, New time = {}'.format(old_time, new_time))

    #check exp map from new and old

    #profile timing for batches of matrices
    #for batch_size in [1, 10, 1000, 10000, 20000]: