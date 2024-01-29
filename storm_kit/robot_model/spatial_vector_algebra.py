"""
Spatial vector algebra
====================================
TODO
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch.profiler import record_function

import math
from storm_kit.differentiable_robot_model import utils
from storm_kit.differentiable_robot_model.utils import vector3_to_skew_symm_matrix
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

    def to_matrix(self)->torch.Tensor:
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

    def to_matrix_transpose(self)->torch.Tensor:
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

@torch.jit.script
class SpatialMotionVec(object):
    lin_motion: torch.Tensor
    ang_motion: torch.Tensor

    def __init__(
        self,
        lin_motion: torch.Tensor = torch.zeros(1,3),
        ang_motion: torch.Tensor = torch.zeros(1,3),
        device: torch.device=torch.device('cpu'),
    ):
        # if lin_motion is None or ang_motion is None:
        #     assert (
        #         device is not None
        #     ), "Cannot initialize with default values without specifying device."
        #     device = torch.device(device)
        self._device = device
        self.lin = lin_motion.to(self._device)
        self.ang = ang_motion.to(self._device)
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
        return SpatialMotionVec(self.lin + smv.lin, self.ang + smv.ang, device=self._device)
    
    def cross_motion_vec(self, smv: SpatialMotionVec) -> SpatialMotionVec:
        r"""
        Args:
            smv: spatial motion vector
        Returns:
            the cross product between motion vectors
        """
        new_ang = cross_product(self.ang, smv.ang)
        new_lin = cross_product(self.ang, smv.lin) + cross_product(self.lin, smv.ang)
        return SpatialMotionVec(new_lin, new_ang, device=self._device)

    # def cross_force_vec(self, sfv: SpatialForceVec) -> SpatialForceVec:
    #     r"""
    #     Args:
    #         sfv: spatial force vector
    #     Returns:
    #         the cross product between motion (self) and force vector
    #     """
    #     new_ang = cross_product(self.ang, sfv.ang) + cross_product(self.lin, sfv.lin)
    #     new_lin = cross_product(self.ang, sfv.lin)
    #     return SpatialForceVec(new_lin, new_ang, device=self._device)

    def transform(self, transform: CoordinateTransform) -> SpatialMotionVec:
        r"""
        Args:
            transform: a coordinate transform object
        Returns:
            the motion vector (self) transformed by the coordinate transform
        """
        new_ang = (transform.rotation() @ self.ang.unsqueeze(2)).squeeze(2)
        with record_function('trans_cross_rot'):
            new_lin = (transform.trans_cross_rot() @ self.ang.unsqueeze(2)).squeeze(2)
        new_lin += (transform.rotation() @ self.lin.unsqueeze(2)).squeeze(2)
        return SpatialMotionVec(new_lin, new_ang, device=self._device)

    def get_vector(self)->torch.Tensor:
        return torch.cat([self.ang, self.lin], dim=1)

    # def multiply(self, v:torch.Tensor)->SpatialForceVec:
    #     batch_size = self.lin.shape[0]
    #     return SpatialForceVec(
    #         self.lin * v.view(batch_size, 1), self.ang * v.view(batch_size, 1),
    #         device=self._device
    #     )

    def dot(self, smv:SpatialMotionVec)->torch.Tensor:
        tmp1 = torch.sum(self.ang * smv.ang, dim=-1)
        tmp2 = torch.sum(self.lin * smv.lin, dim=-1)
        return tmp1 + tmp2

class SpatialForceVec(object):
    lin_force: torch.Tensor
    ang_force: torch.Tensor
    def __init__(
        self,
        lin_force: torch.Tensor = torch.zeros(1,3),
        ang_force: torch.Tensor = torch.zeros(1,3),
        device:torch.device = torch.device('cpu'),
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

    def multiply(self, v:torch.Tensor) -> SpatialForceVec:
        batch_size = self.lin.shape[0]
        return SpatialForceVec(
            self.lin * v.view(batch_size, 1), self.ang * v.view(batch_size, 1),
            device=self.device
        )

    def dot(self, smv:SpatialMotionVec)->torch.Tensor:
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
def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


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
def y_rot(angle: torch.Tensor) -> torch.Tensor:
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
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

@torch.jit.script
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

@torch.jit.script
def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

@torch.jit.script
def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


@torch.jit.script
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


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


@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4). [qw, qx,qy,qz]
    """
    # if matrix.size(-1) != 3 or matrix.size(-2) != 3:
    #     raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    zero = matrix.new_zeros((1,))
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]
    qw = 0.5 * torch.sqrt(torch.max(zero, 1.0 + m00 + m11 + m22))
    qx = 0.5 * torch.sqrt(torch.max(zero, 1.0 + m00 - m11 - m22))
    qy = 0.5 * torch.sqrt(torch.max(zero, 1.0 - m00 + m11 - m22))
    qz = 0.5 * torch.sqrt(torch.max(zero, 1.0 - m00 - m11 + m22))

    cond1 = ((qw >= qx) & (qw >= qy) & (qw >= qz))
    cond2 = (~cond1 & (qx >= qw) & (qx >= qy) & (qx >= qz))
    cond3 = (~cond1 & ~cond2 & (qy >= qw) & (qy >= qx) & (qy >= qz))
    cond4 = (~cond1 & ~cond2 & ~cond3 & (qz >= qw) & (qz >= qx) & (qz >= qy))
    cond1 = cond1.nonzero()
    cond2 = cond2.nonzero()
    cond3 = cond3.nonzero()
    cond4 = cond4.nonzero()
    
    #When qw is max
    qw[cond1] *= 1.0
    qx[cond1] = _copysign(qx[cond1], m21[cond1] - m12[cond1])
    qy[cond1] = _copysign(qy[cond1], m02[cond1] - m20[cond1])
    qz[cond1] = _copysign(qz[cond1], m10[cond1] - m01[cond1])
    
    #When qx is max
    qw[cond2] = _copysign(qw[cond2], m21[cond2] - m12[cond2])
    qx[cond2] *= 1.0
    qy[cond2] = _copysign(qy[cond2], m10[cond2] + m01[cond2])
    qz[cond2] = _copysign(qz[cond2], m02[cond2] + m20[cond2])

    #When qy is max
    qw[cond3] = _copysign(qw[cond3], m02[cond3] - m20[cond3])
    qx[cond3] = _copysign(qx[cond3], m10[cond3] + m01[cond3])
    qy[cond3] *= 1.0
    qz[cond3] = _copysign(qz[cond3], m21[cond3] + m12[cond3])
    
    #When qz is max
    qw[cond4] = _copysign(qw[cond4], m10[cond4] - m01[cond4])
    qx[cond4] = _copysign(qx[cond4], m20[cond4] + m02[cond4])
    qy[cond4] = _copysign(qy[cond4], m21[cond4] + m12[cond4])
    qz[cond4] *= 1.0

    q = torch.stack((qw, qx, qy, qz), -1)
    q /= torch.norm(q, dim=-1, p=2)[...,None]

    return q

@torch.jit.script
def matrix_to_quaternion2(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    # if matrix.size(-1) != 3 or matrix.size(-2) != 3:
    #     raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


# def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
#     """
#     Convert rotations given as rotation matrices to quaternions.

#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).

#     Returns:
#         quaternions with real part first, as tensor of shape (..., 4). [qw, qx,qy,qz]
#     """
#     # if matrix.size(-1) != 3 or matrix.size(-2) != 3:
#     #     raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
#     zero = matrix.new_zeros((1,))
#     m00 = matrix[..., 0, 0]
#     m11 = matrix[..., 1, 1]
#     m22 = matrix[..., 2, 2]
#     o0 = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 + m11 + m22))
#     x = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 - m11 - m22))
#     y = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 + m11 - m22))
#     z = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 - m11 + m22))
#     o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
#     o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
#     o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
#     return torch.stack((o0, o1, o2, o3), -1)

@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor)->torch.Tensor:
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

# def quaternion_to_matrix2(quaternions: torch.Tensor)->torch.Tensor:
#     """
#     Convert rotations given as quaternions to rotation matrices.

#     Args:
#         quaternions: quaternions with real part first,
#             as tensor of shape (..., 4).

#     Returns:
#         Rotation matrices as tensor of shape (..., 3, 3).
#     """
#     w, x, y, z = torch.unbind(quaternions, -1)
#     # two_s = 2.0 / (quaternions * quaternions).sum(-1)
#     tx = 2*x
#     ty = 2*y
#     tz = 2*z
#     # xx = tx * x
#     # yy = ty * y
#     # zz = tz * z
#     xy = ty * x
#     xz = tz * x
#     yz = ty * z
#     wx = tx * w
#     wy = ty * w
#     wz = tz * w

#     # diagonal terms
#     t0 = (w+y)*(w-y); t1 = (x+z)*(x-z)
#     t2 = (w+x)*(w-x); t3 = (y+z)*(y-z)
#     m00 = t0+t1
#     m11 = t2+t3
#     m22 = t2-t3
#     # m00 = 1.0 - (yy + zz) 
#     # m11 = 1.0 - (xx + zz)
#     # m22 = 1.0 - (xx + yy)

#     m10 = xy + wz 
#     m01 = xy - wz
#     m20 = xz - wy
#     m02 = xz + wy
#     m21 = yz + wx
#     m12 = yz - wx

#     row1 = torch.stack((m00, m01, m02), -1)
#     row2 = torch.stack((m10, m11, m12), -1)
#     row3 = torch.stack((m20, m21, m22), -1)
#     m = torch.stack((row1, row2, row3), -1)
#     return m


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

@torch.jit.script
def quat_multiply(q1:torch.Tensor, q2:torch.Tensor) -> torch.Tensor:
    a_w = q1[..., 0]
    a_x = q1[..., 1]
    a_y = q1[..., 2]
    a_z = q1[..., 3]
    b_w = q2[..., 0]
    b_x = q2[..., 1]
    b_y = q2[..., 2]
    b_z = q2[..., 3]


    q_res_w = (a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z).unsqueeze(-1)
    q_res_x = (a_w * b_x + b_w * a_x + a_y * b_z - b_y * a_z).unsqueeze(-1)
    q_res_y = (a_w * b_y + b_w * a_y + a_z * b_x - b_z * a_x).unsqueeze(-1)
    q_res_z = (a_w * b_z + b_w * a_z + a_x * b_y - b_x * a_y).unsqueeze(-1)

    return torch.cat([q_res_w, q_res_x, q_res_y, q_res_z], dim=-1)




if __name__ == "__main__":
    # Test 1: Convert [0, 1, 0, 0] to rot and convert back
    # Test 2: Convert [0, 0.7071, 0.7071, 0] to rot and convert back
    # Test 3: Convert [0, -0.7071, -0.7071, 0] to rot and convert back
    # Test 4: Convert [-1, 0, 0, 0] to rot and convert back
    quat1 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0)
    quat2 = torch.tensor([0.0, 0.7071, 0.7071, 0.0], dtype=torch.float32).unsqueeze(0)
    quat3 = torch.tensor([0.0, -0.7071, -0.7071, 0.0], dtype=torch.float32).unsqueeze(0)
    quat4 = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0)

    assert torch.allclose(matrix_to_quaternion(quaternion_to_matrix(quat1)), quat1)
    assert torch.allclose(matrix_to_quaternion(quaternion_to_matrix(quat2)), quat2)
    assert torch.allclose(matrix_to_quaternion(quaternion_to_matrix(quat3)), quat3)
    assert torch.allclose(matrix_to_quaternion(quaternion_to_matrix(quat4)), quat4)