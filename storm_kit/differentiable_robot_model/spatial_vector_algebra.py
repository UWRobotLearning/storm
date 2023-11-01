"""
Spatial vector algebra
====================================
TODO
"""
from __future__ import annotations
from typing import Optional
import torch
import math
from storm_kit.differentiable_robot_model import utils
from .utils import vector3_to_skew_symm_matrix
from storm_kit.differentiable_robot_model.utils import cross_product

@torch.jit.script
class CoordinateTransform(object):
    rot: torch.Tensor
    trans: torch.Tensor

    def __init__(self, 
                rot: torch.Tensor = torch.eye(3).unsqueeze(0), 
                trans: torch.Tensor = torch.zeros(1,3), 
                device: torch.device=torch.device("cpu")):
        
        self._device = device

        # if rot is None:
        #     self._rot = torch.eye(3, device=self._device)
        # else:
        self._rot = rot.to(self._device)
        # if len(self._rot.shape) == 2:
        #     self._rot = self._rot.unsqueeze(0)

        # if trans is None:
        #     self._trans = torch.zeros(3, device=self._device)
        # else:
        self._trans = trans.to(self._device)
        # if len(self._trans.shape) == 1:
        #     self._trans = self._trans.unsqueeze(0)

    def set_translation(self, t:torch.Tensor)->None:
        self._trans = t.to(self._device)
        # if len(self._trans.shape) == 1:
        #     self._trans = self._trans.unsqueeze(0)
        return

    def set_rotation(self, rot:torch.Tensor)->None:
        self._rot = rot.to(self._device)
        # if len(self._rot.shape) == 2:
        #     self._rot = self._rot.unsqueeze(0)
        return

    def rotation(self)-> torch.Tensor:
        return self._rot

    def translation(self)->torch.Tensor:
        return self._trans

    def inverse(self)->CoordinateTransform:
        rot_transpose = self._rot.transpose(-2, -1)
        return CoordinateTransform(
            rot_transpose, -(rot_transpose @ self._trans.unsqueeze(2)).squeeze(2),
            device=self._device
        )

    def multiply_transform(self, coordinate_transform:CoordinateTransform)->CoordinateTransform:
        new_rot = self._rot @ coordinate_transform.rotation()
        new_trans = (
            self._rot @ coordinate_transform.translation().unsqueeze(-1)
        ).squeeze(-1) + self._trans
        return CoordinateTransform(new_rot, new_trans, device=self._device)

    def trans_cross_rot(self)->torch.Tensor:
        return vector3_to_skew_symm_matrix(self._trans) @ self._rot

    def get_quaternion(self)->torch.Tensor:
        #TODO: Ust jit function
        batch_size = self._rot.shape[0]
        M = torch.zeros((batch_size, 4, 4)).to(self._rot.device)
        M[:, :3, :3] = self._rot
        M[:, :3, 3] = self._trans
        M[:, 3, 3] = 1
        q = torch.empty((batch_size, 4)).to(self._rot.device)
        t = torch.einsum("bii->b", M)  # torch.trace(M)
        for n in range(batch_size):
            tn = t[n]
            if tn > M[n, 3, 3]:
                q[n, 3] = tn
                q[n, 2] = M[n, 1, 0] - M[n, 0, 1]
                q[n, 1] = M[n, 0, 2] - M[n, 2, 0]
                q[n, 0] = M[n, 2, 1] - M[n, 1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[n, 1, 1] > M[n, 0, 0]:
                    i, j, k = 1, 2, 0
                if M[n, 2, 2] > M[n, i, i]:
                    i, j, k = 2, 0, 1
                tn = M[n, i, i] - (M[n, j, j] + M[n, k, k]) + M[n, 3, 3]
                q[n, i] = tn
                q[n, j] = M[n, i, j] + M[n, j, i]
                q[n, k] = M[n, k, i] + M[n, i, k]
                q[n, 3] = M[n, k, j] - M[n, j, k]
                # q = q[[3, 0, 1, 2]]
            q[n, :] *= 0.5 / math.sqrt(tn * M[n, 3, 3])
        return q

    def to_matrix(self):
        batch_size = self._rot.shape[0]

        mat = torch.zeros((batch_size, 6, 6), device=self._device)
        t = torch.zeros((batch_size, 3, 3), device=self._device)
        t[:, 0, 1] = -self._trans[:, 2]
        t[:, 0, 2] = self._trans[:, 1]
        t[:, 1, 0] = self._trans[:, 2]
        t[:, 1, 2] = -self._trans[:, 0]
        t[:, 2, 0] = -self._trans[:, 1]
        t[:, 2, 1] = self._trans[:, 0]
        _Erx = self._rot.transpose(-2, -1).matmul(t)

        mat[:, :3, :3] = self._rot.transpose(-2, -1)
        mat[:, 3:, 0:3] = -_Erx
        mat[:, 3:, 3:] = self._rot.transpose(-2, -1)
        return mat

    def to_matrix_transpose(self):
        batch_size = self._rot.shape[0]

        mat = torch.zeros((batch_size, 6, 6), device=self._device)
        t = torch.zeros((batch_size, 3, 3), device=self._device)
        t[:, 0, 1] = -self._trans[:, 2]
        t[:, 0, 2] = self._trans[:, 1]
        t[:, 1, 0] = self._trans[:, 2]
        t[:, 1, 2] = -self._trans[:, 0]
        t[:, 2, 0] = -self._trans[:, 1]
        t[:, 2, 1] = self._trans[:, 0]
        _Erx = self._rot.matmul(t)

        mat[:, :3, :3] = self._rot.transpose(-1, -2)
        mat[:, 3:, 0:3] = -_Erx.transpose(-1, -2)
        mat[:, 3:, 3:] = self._rot.transpose(-1, -2)
        return mat

class SpatialMotionVec(object):
    def __init__(
        self,
        lin_motion: torch.Tensor = torch.zeros(1,3),
        ang_motion: torch.Tensor = torch.zeros(1,3),
        device:torch.device = torch.device('cpu'),
    ):
        # if lin_motion is None or ang_motion is None:
        #     assert (
        #         device is not None
        #     ), "Cannot initialize with default values without specifying device."
        #     device = torch.device(device)
        self.device = device
        self.lin = lin_motion.to(self.device)
        self.ang = ang_motion.to(self.device)
        # self.lin = (
        #     lin_motion if lin_motion is not None else torch.zeros((1, 3), device=device)
        # )
        # self.ang = (
        #     ang_motion if ang_motion is not None else torch.zeros((1, 3), device=device)
        # )

    def add_motion_vec(self, smv: SpatialMotionVec) -> SpatialMotionVec:
        r"""
        Args:
            smv: spatial motion vector
        Returns:
            the sum of motion vectors
        """
        return SpatialMotionVec(self.lin + smv.lin, self.ang + smv.ang, device=self.device)

    def cross_motion_vec(self, smv: SpatialMotionVec) -> SpatialMotionVec:
        r"""
        Args:
            smv: spatial motion vector
        Returns:
            the cross product between motion vectors
        """
        new_ang = cross_product(self.ang, smv.ang)
        new_lin = cross_product(self.ang, smv.lin) + cross_product(self.lin, smv.ang)
        return SpatialMotionVec(new_lin, new_ang, device=self.device)

    def cross_force_vec(self, sfv: SpatialForceVec) -> SpatialForceVec:
        r"""
        Args:
            sfv: spatial force vector
        Returns:
            the cross product between motion (self) and force vector
        """
        new_ang = cross_product(self.ang, sfv.ang) + cross_product(self.lin, sfv.lin)
        new_lin = cross_product(self.ang, sfv.lin)
        return SpatialForceVec(new_lin, new_ang, device=self.device)

    def transform(self, transform: CoordinateTransform) -> SpatialMotionVec:
        r"""
        Args:
            transform: a coordinate transform object
        Returns:
            the motion vector (self) transformed by the coordinate transform
        """
        new_ang = (transform.rotation() @ self.ang.unsqueeze(2)).squeeze(2)
        new_lin = (transform.trans_cross_rot() @ self.ang.unsqueeze(2)).squeeze(2)
        new_lin += (transform.rotation() @ self.lin.unsqueeze(2)).squeeze(2)
        return SpatialMotionVec(new_lin, new_ang, device=self.device)

    def get_vector(self)->torch.Tensor:
        return torch.cat([self.ang, self.lin], dim=1)

    def multiply(self, v:torch.Tensor)->SpatialForceVec:
        batch_size = self.lin.shape[0]
        return SpatialForceVec(
            self.lin * v.view(batch_size, 1), self.ang * v.view(batch_size, 1),
            device=self.device
        )

    def dot(self, smv:SpatialMotionVec)->torch.Tensor:
        tmp1 = torch.sum(self.ang * smv.ang, dim=-1)
        tmp2 = torch.sum(self.lin * smv.lin, dim=-1)
        return tmp1 + tmp2

class SpatialForceVec(object):
    def __init__(
        self,
        lin_force: torch.Tensor = torch.zeros(1,3),
        ang_force: torch.Tensor = torch.zeros(1,3),
        device=None,
    ):
        # if lin_force is None or ang_force is None:
        #     assert (
        #         device is not None
        #     ), "Cannot initialize with default values without specifying device."
        #     device = torch.device(device)
        self.device = device
        self.lin = lin_force.to(self.device)
        self.ang = ang_force.to(self.device)
        # self.lin = (
        #     lin_force if lin_force is not None else torch.zeros((1, 3), device=device)
        # )
        # self.ang = (
        #     ang_force if ang_force is not None else torch.zeros((1, 3), device=device)
        # )

    def add_force_vec(self, sfv: SpatialForceVec) -> SpatialForceVec:
        r"""
        Args:
            sfv: spatial force vector
        Returns:
            the sum of force vectors
        """
        return SpatialForceVec(self.lin + sfv.lin, self.ang + sfv.ang, device=self.device)

    def transform(self, transform: CoordinateTransform) -> SpatialForceVec:
        r"""
        Args:
            transform: a coordinate transform object
        Returns:
            the force vector (self) transformed by the coordinate transform
        """
        new_lin = (transform.rotation() @ self.lin.unsqueeze(2)).squeeze(2)
        new_ang = (transform.trans_cross_rot() @ self.lin.unsqueeze(2)).squeeze(2)
        new_ang += (transform.rotation() @ self.ang.unsqueeze(2)).squeeze(2)
        return SpatialForceVec(new_lin, new_ang, device=self.device)

    def get_vector(self):
        return torch.cat([self.ang, self.lin], dim=1)

    def multiply(self, v:torch.Tensor):
        batch_size = self.lin.shape[0]
        return SpatialForceVec(
            self.lin * v.view(batch_size, 1), self.ang * v.view(batch_size, 1),
            device=self.device
        )

    def dot(self, smv)->torch.Tensor:
        tmp1 = torch.sum(self.ang * smv.ang, dim=-1)
        tmp2 = torch.sum(self.lin * smv.lin, dim=-1)
        return tmp1 + tmp2


class DifferentiableSpatialRigidBodyInertia(torch.nn.Module):
    def __init__(self, rigid_body_params, device="cpu"):
        super().__init__()
        # lambda functions are a "hack" to make this compatible with the learnable variants
        self.mass = lambda: rigid_body_params["mass"]
        self.com = lambda: rigid_body_params["com"]
        self.inertia_mat = lambda: rigid_body_params["inertia_mat"]

        self._device = torch.device(device)

    def _get_parameter_values(self):
        return self.mass(), self.com(), self.inertia_mat()

    def multiply_motion_vec(self, smv)->SpatialForceVec:
        mass, com, inertia_mat = self._get_parameter_values()
        mcom = com * mass
        com_skew_symm_mat = vector3_to_skew_symm_matrix(com)
        inertia = inertia_mat + mass * (
            com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
        )

        batch_size = smv.lin.shape[0]

        new_lin_force = mass * smv.lin - utils.cross_product(
            mcom.repeat(batch_size, 1), smv.ang
        )
        new_ang_force = (
            inertia.repeat(batch_size, 1, 1) @ smv.ang.unsqueeze(2)
        ).squeeze(2) + utils.cross_product(mcom.repeat(batch_size, 1), smv.lin)

        return SpatialForceVec(new_lin_force, new_ang_force)

    def get_spatial_mat(self)->torch.Tensor:
        mass, com, inertia_mat = self._get_parameter_values()
        mcom = mass * com
        com_skew_symm_mat = vector3_to_skew_symm_matrix(com)
        inertia = inertia_mat + mass * (
            com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
        )
        mat = torch.zeros((6, 6), device=self._device)
        mat[:3, :3] = inertia
        mat[3, 0] = 0
        mat[3, 1] = mcom[0, 2]
        mat[3, 2] = -mcom[0, 1]
        mat[4, 0] = -mcom[0, 2]
        mat[4, 1] = 0.0
        mat[4, 2] = mcom[0, 0]
        mat[5, 0] = mcom[0, 1]
        mat[5, 1] = -mcom[0, 0]
        mat[5, 2] = 0.0

        mat[0, 3] = 0
        mat[0, 4] = -mcom[0, 2]
        mat[0, 5] = mcom[0, 1]
        mat[1, 3] = mcom[0, 2]
        mat[1, 4] = 0.0
        mat[1, 5] = -mcom[0, 0]
        mat[2, 3] = -mcom[0, 1]
        mat[2, 4] = mcom[0, 0]
        mat[2, 5] = 0.0

        mat[3, 3] = mass
        mat[4, 4] = mass
        mat[5, 5] = mass
        return mat


#torch JIT compatible functions
@torch.jit.script
def _copysign(a: torch.Tensor, b: torch.Tensor):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

@torch.jit.script
def x_rot(angle: torch.Tensor) -> torch.Tensor:
    #  if len(angle.shape) == 0:
    # angle = angle.unsqueeze(0)
    # angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device, dtype=angle.dtype)
    R[:, 0, 0] = torch.ones(batch_size, device=angle.device, dtype=angle.dtype)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 1, 2] = -torch.sin(angle)
    R[:, 2, 1] = torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R

@torch.jit.script
def y_rot(angle: torch.Tensor)->torch.Tensor:
    #  if len(angle.shape) == 0:
    # angle = angle.unsqueeze(0)
    # angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device, dtype=angle.dtype)
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 2] = torch.sin(angle)
    R[:, 1, 1] = torch.ones(batch_size, device=angle.device, dtype=angle.dtype)
    R[:, 2, 0] = -torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R

@torch.jit.script
def z_rot(angle: torch.Tensor) -> torch.Tensor:
    #  if len(angle.shape) == 0:
    # angle = angle.unsqueeze(0)
    # angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device, dtype=angle.dtype)
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 1] = -torch.sin(angle)
    R[:, 1, 0] = torch.sin(angle)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 2, 2] = torch.ones(batch_size, device=angle.device, dtype=angle.dtype)
    return R



@torch.jit.script
def matrix_to_euler_angles(R: torch.Tensor, cy_thresh: float = 1e-6):
    # if cy_thresh is None:
    #     try:
    #         cy_thresh = np.finfo(M.dtype).eps * 4
    #     except ValueError:
    #         cy_thresh = _FLOAT_EPS_4
    # r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    inp_device = R.device
    #if(len(R.shape) == 4):
    Z = torch.zeros(R.shape[:-2], device=inp_device, dtype=R.dtype)
    #print(Z.shape)
    #else:
    #    Z = torch.zeros(R.shape[0], device=inp_device, dtype=R.dtype)
    r11 = R[...,0,0]
    r12 = R[...,0,1]
    r13 = R[...,0,2]
    r21 = R[...,1,0]
    r22 = R[...,1,1]
    r23 = R[...,1,2]
    r31 = R[...,2,0]
    r32 = R[...,2,1]
    r33 = R[...,2,2]


    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = torch.sqrt(r33*r33 + r23*r23)

    cond = cy > cy_thresh

    z = torch.where(cond, torch.atan2(-r12,  r11), torch.atan2(r21,  r22)).unsqueeze(-1)
    y = torch.atan2(r13,  cy).unsqueeze(-1)
    x = torch.where(cond, torch.atan2(-r23, r33), Z).unsqueeze(-1) 

    # if cy > cy_thresh: # cos(y) not close to zero, standard form
    #     z = torch.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
    #     y = torch.atan2(r13,  cy) # atan2(sin(y), cy)
    #     x = torch.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    # else: # cos(y) (close to) zero, so x -> 0.0 (see above)
    #     # so r21 -> sin(z), r22 -> cos(z) and
    #     z = torch.atan2(r21,  r22)
    #     y = torch.atan2(r13,  cy) # atan2(sin(y), cy)
    #     x = 0.0
    
    # return z, y, x
    return torch.cat([x, y, z], dim=-1)

@torch.jit.script
def rpy_angles_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as RPY euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    roll = euler_angles[:,0]
    pitch = euler_angles[:,1]
    yaw = euler_angles[:,2]
    matrices = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

    return matrices


#[qw, qx, qy, qz]
# Test 1: Convert [0, 1, 0, 0] to rot and convert back
# Test 2: Convert [-1, 0, 0, 0] to rot and convert back
# TODO: Test also quaternion_to_matrix
@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4). [qw, qx,qy,qz]
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    zero = matrix.new_zeros((1,))
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 + m11 + m22))
    x = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 - m11 - m22))
    y = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 + m11 - m22))
    z = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 - m11 + m22))
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))



@torch.jit.script   
def multiply_transform(w_rot_l: torch.Tensor, w_trans_l: torch.Tensor, l_rot_c: torch.Tensor, l_trans_c: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    w_rot_c = w_rot_l @ l_rot_c    
    w_trans_c = (w_rot_l @ l_trans_c.unsqueeze(-1)).squeeze(-1) + w_trans_l
    return w_rot_c, w_trans_c

@torch.jit.script   
def multiply_inv_transform(l_rot_w: torch.Tensor, l_trans_w: torch.Tensor, l_rot_c: torch.Tensor, l_trans_c: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    w_rot_l = l_rot_w.transpose(-1,-2)
    w_rot_c = w_rot_l @ l_rot_c


    w_trans_l = -(w_rot_l @ l_trans_w.unsqueeze(2)).squeeze(2)
    w_trans_c = (w_rot_l @ l_trans_c.unsqueeze(-1)).squeeze(-1) + w_trans_l

    return w_rot_c, w_trans_c


@torch.jit.script
def transform_point(point: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor)->torch.Tensor:
    #new_point = (rot @ (point).unsqueeze(-1)).squeeze(-1) + trans
    new_point = (point @ rot.transpose(-1,-2)) + trans
    return new_point