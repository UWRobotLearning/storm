# Copyright (c) Facebook, Inc. and its affiliates.
"""
Differentiable rigid body
====================================
TODO
"""

from typing import List, Optional, Dict

import torch
from .spatial_vector_algebra import (
    CoordinateTransform,
    z_rot,
    y_rot,
    x_rot,
)

from .spatial_vector_algebra import SpatialForceVec, SpatialMotionVec
from .spatial_vector_algebra import (
    DifferentiableSpatialRigidBodyInertia,
)


class DifferentiableRigidBody(torch.nn.Module):
    """
    Differentiable Representation of a link
    """

    joint_limits: Dict[str, float]
    name: str

    # _parents: Optional["DifferentiableRigidBody"]
    # _children: List["DifferentiableRigidBody"]

    def __init__(self, rigid_body_params,
                 device: torch.device = torch.device('cpu')):

        super().__init__()

        # self._parents = None
        # self._children = []

        self._device = device
        self.joint_id = rigid_body_params["joint_id"]
        self.name:str = rigid_body_params["link_name"]

        # parameters that can be made learnable
        self.inertia = DifferentiableSpatialRigidBodyInertia(
            rigid_body_params, device=self._device
        )
        self.joint_damping = rigid_body_params["joint_damping"]
        self.trans = rigid_body_params["trans"]#.reshape(1, 3)
        self.rot_angles = rigid_body_params["rot_angles"].unsqueeze(0)#.reshape(1, 3)
        # end parameters that can be made learnable
        
        # rot_angle_vals = self.rot_angles #()
        # roll = rot_angle_vals[:,0]
        # pitch = rot_angle_vals[:,1]
        # yaw = rot_angle_vals[:,2]

        roll = self.rot_angles[:,0]
        pitch = self.rot_angles[:,1]
        yaw = self.rot_angles[:,2]

        self.fixed_rotation = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

        # local joint axis (w.r.t. joint coordinate frame):
        self.joint_axis = rigid_body_params["joint_axis"]

        # if torch.abs(self.joint_axis[0, 0]) == 1:
        #     rot = x_rot(torch.sign(self.joint_axis[0, 0]) * q)
        # elif torch.abs(self.joint_axis[0, 1]) == 1:
        #     rot = y_rot(torch.sign(self.joint_axis[0, 1]) * q)
        # else:
        #     rot = z_rot(torch.sign(self.joint_axis[0, 2]) * q)
        if self.joint_axis[0, 0] == 1:
            self.axis_rot_fn = x_rot
        elif self.joint_axis[0, 1] == 1:
            self.axis_rot_fn = y_rot
        else:
            self.axis_rot_fn = z_rot

        self.joint_limits = {'effort': torch.inf, 'lower': -torch.inf, 'upper': torch.inf, 'velocity': -torch.inf}
        if rigid_body_params["joint_limits"] is not None:
            self.joint_limits = rigid_body_params["joint_limits"]

        self.joint_pose = CoordinateTransform(device=self._device)
        self.joint_pose.set_translation(torch.reshape(self.trans, (1, 3)))

        # local velocities and accelerations (w.r.t. joint coordinate frame):
        self.joint_vel = SpatialMotionVec(device=self._device)
        self.joint_acc = SpatialMotionVec(device=self._device)

        self._batch_size = -1
        self._batch_trans = None
        self._batch_rot = None

        self.update_joint_state(
            torch.zeros([1, 1], device=self._device),
            torch.zeros([1, 1], device=self._device),
        )
        self.update_joint_acc(torch.zeros([1, 1], device=self._device))

        self.pose = CoordinateTransform(device=self._device)

        self.vel = SpatialMotionVec(device=self._device)
        self.acc = SpatialMotionVec(device=self._device)

        self.force = SpatialForceVec(device=self._device)

    # # Kinematic tree construction
    # @torch.jit.export
    # def set_parent(self, link: "DifferentiableRigidBody"):
    #     self._parent = link
    
    # @torch.jit.export
    # def add_child(self, link: "DifferentiableRigidBody"):
    #     self._children.append(link)

    # # Recursive algorithms
    # def forward_kinematics(self, q_dict):
    #     """Recursive forward kinematics
    #     Computes transformations from self to all descendants.

    #     Returns: Dict[link_name, transform_from_self_to_link]
    #     """
    #     # Compute joint pose
    #     if self.name in q_dict:
    #         q = q_dict[self.name]
    #         batch_size = q.shape[0]

    #         rot_angles_vals = self.rot_angles()
    #         roll = rot_angles_vals[0, 0]
    #         pitch = rot_angles_vals[0, 1]
    #         yaw = rot_angles_vals[0, 2]
    #         fixed_rotation = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

    #         if torch.abs(self.joint_axis[0, 0]) == 1:
    #             rot = x_rot(torch.sign(self.joint_axis[0, 0]) * q)
    #         elif torch.abs(self.joint_axis[0, 1]) == 1:
    #             rot = y_rot(torch.sign(self.joint_axis[0, 1]) * q)
    #         else:
    #             rot = z_rot(torch.sign(self.joint_axis[0, 2]) * q)

    #         joint_pose = CoordinateTransform(
    #             rot=fixed_rotation.repeat(batch_size, 1, 1) @ rot,
    #             trans=torch.reshape(self.trans(), (1, 3)).repeat(batch_size, 1),
    #             device=self._device,
    #         )

    #     else:
    #         joint_pose = self.joint_pose

    #     # Compute forward kinematics of children
    #     pose_dict = {self.name: self.pose}
    #     for child in self._children:
    #         pose_dict.update(child.forward_kinematics(q_dict))

    #     # Apply joint pose
    #     return {
    #         body_name: joint_pose.multiply_transform(pose_dict[body_name])
    #         for body_name in pose_dict
    #     }

    # Get/set
    @torch.jit.export
    def update_joint_state(self, q:torch.Tensor, qd:torch.Tensor):
        batch_size = q.shape[0]

        joint_ang_vel = qd @ self.joint_axis
        self.joint_vel = SpatialMotionVec(
            ang_motion=joint_ang_vel,
            device=self._device
        )

        if batch_size != self._batch_size:
            #print('calling once', self._batch_size, batch_size)
            self._batch_size = batch_size
            self._batch_trans = self.trans.repeat(self._batch_size,1)
            self._batch_rot = self.fixed_rotation.repeat(self._batch_size, 1, 1)


        # rot_angles_vals = self.rot_angles()
        # roll = rot_angles_vals[:, 0]
        # pitch = rot_angles_vals[:, 1]
        # yaw = rot_angles_vals[:, 2]

        # fixed_rotation = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

        # when we update the joint angle, we also need to update the transformation
        # self.joint_pose.set_translation(
        #     torch.reshape(self.trans(), (1, 3)).repeat(batch_size, 1)
        # )
        self.joint_pose.set_translation(self._batch_trans)#

        # if torch.abs(self.joint_axis[0, 0]) == 1:
        #     rot = x_rot(torch.sign(self.joint_axis[0, 0]) * q)
        # elif torch.abs(self.joint_axis[0, 1]) == 1:
        #     rot = y_rot(torch.sign(self.joint_axis[0, 1]) * q)
        # else:
        #     rot = z_rot(torch.sign(self.joint_axis[0, 2]) * q)
        rot = self.axis_rot_fn(q.squeeze(1))
        # self.joint_pose.set_rotation(fixed_rotation.repeat(batch_size, 1, 1) @ rot)
        self.joint_pose.set_rotation(self._batch_rot @ rot)
        return
    
    @torch.jit.export
    def update_joint_acc(self, qdd:torch.Tensor):
        # local z axis (w.r.t. joint coordinate frame):
        joint_ang_acc = qdd @ self.joint_axis
        self.joint_acc = SpatialMotionVec(
            torch.zeros_like(joint_ang_acc), joint_ang_acc,
            device=self._device
        )
        return

    @torch.jit.export
    def get_joint_limits(self):
        return self.joint_limits

    @torch.jit.export
    def get_joint_damping_const(self):
        return self.joint_damping
