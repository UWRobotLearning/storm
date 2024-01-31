# Copyright (c) Facebook, Inc. and its affiliates.
"""
Differentiable robot model class
====================================
TODO
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
from torch.profiler import record_function

from storm_kit.differentiable_robot_model.spatial_vector_algebra import SpatialMotionVec, SpatialForceVec
from storm_kit.util_file import join_path, get_assets_path
from .spatial_vector_algebra import SpatialForceVec, SpatialMotionVec, DifferentiableSpatialRigidBodyInertia
from .spatial_vector_algebra import (
    CoordinateTransform,
    z_rot,
    y_rot,
    x_rot,
)
from .urdf_utils import URDFRobotModel


class RigidBody(nn.Module):
    """
    Differentiable Representation of a link
    """

    joint_limits: Dict[str, float]
    name: str
    _batch_rot: torch.Tensor
    _batch_trans: torch.Tensor
    # _children: List[RigidBody]

    # _parents: Optional["RigidBody"]
    # _children: List["RigidBody"]

    def __init__(self, rigid_body_params,
                 device: torch.device = torch.device('cpu')):

        super().__init__()

        # self._parents = None
        self._children: List[RigidBody] = []

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
    # def set_parent(self, link: "RigidBody"):
    #     self._parent = link
    
    # @torch.jit.export
    # def add_child(self, link: "RigidBody"):
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


class RobotModel(nn.Module):
    """
    Robot Model
    ====================================
    """
    urdf: str
    mesh_dir: str
    _name_to_idx_map: Dict[str, int]
    _body_idx_to_controlled_joint_idx_map: Dict[int, int]
    _name_to_parent_map: Dict[str, str]
    _body_to_joint_idx_map: Dict[str, int]
    _bodies: List[RigidBody]
    _joint_limits: List[Dict[str, float]]
    _link_names: List[str]
    _link_pose_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]

    def __init__(self, robot_config, device:torch.device=torch.device('cpu')):
        super().__init__()
        self.config = robot_config
        self.name = self.config['name']
        self._device = device
        self.urdf:str = join_path(get_assets_path(), self.config['urdf_path'])
        self.mesh_dir:str = join_path(get_assets_path(), self.config['mesh_dir'])
        self._urdf_model = URDFRobotModel(urdf_path=self.urdf, device=self._device)
        self._bodies = [] #torch.nn.ModuleList()
        self._n_dofs = 0
        self._controlled_joints = []

        self._batch_size:int = 1
        self._base_lin_vel:torch.Tensor = torch.zeros((self._batch_size, 3), device=self._device) #, dtype=self.dtype)
        self._base_ang_vel:torch.Tensor = torch.zeros((self._batch_size, 3), device=self._device) #, dtype=self.dtype)
        self._base_pose_trans:torch.Tensor = torch.zeros((self._batch_size, 3), device=self._device) #, dtype=self.dtype)
        self._base_pose_rot:torch.Tensor = torch.eye(3, device=self._device).expand(self._batch_size,3,3) #, dtype=self.dtype)

        # here we're making the joint a part of the rigid body
        # while urdfs model joints and rigid bodies separately
        # joint is at the beginning of a link
        self._name_to_idx_map = dict()
        self._body_idx_to_controlled_joint_idx_map = dict()
        self._name_to_parent_map = dict()
        self._body_to_joint_idx_map = dict()
        controlled_jnt_idx = 0


        for (i, link) in enumerate(self._urdf_model.robot.links):
            # Initialize body object
            rigid_body_params = self._urdf_model.get_body_parameters_from_urdf(i, link)
            body = torch.jit.script(RigidBody(
                rigid_body_params=rigid_body_params, device=self._device
            ))

            # Joint properties
            body.joint_idx = None
            if rigid_body_params["joint_type"] != "fixed":
                body.joint_idx = self._n_dofs
                self._n_dofs += 1
                self._controlled_joints.append(i)
                self._body_idx_to_controlled_joint_idx_map[i] = controlled_jnt_idx
                controlled_jnt_idx += 1

            # Add to data structures
            self._bodies.append(body)
            self._name_to_idx_map[body.name] = i
            self._name_to_parent_map[body.name] = self._urdf_model.get_name_of_parent_body(body.name)
            self._body_to_joint_idx_map[body.name] = self._urdf_model.find_joint_of_body(body.name)

        self._joint_limits = []
        for idx in self._controlled_joints:
            self._joint_limits.append(self._bodies[idx].get_joint_limits())

        self._link_names = []
        for i in range(len(self._bodies)):
            self._link_names.append(self._bodies[i].name)

        # Once all bodies are loaded, connect each body to its parent
        # for body in self._bodies[1:]:
        #     parent_body_name = self._urdf_model.get_name_of_parent_body(body.name)
        #     parent_body_idx = self._name_to_idx_map[parent_body_name]
        #     body.set_parent(self._bodies[parent_body_idx])
        #     self._bodies[parent_body_idx].add_child(body)

        self._lin_jac, self._ang_jac = (
            torch.zeros([self._batch_size, 3, self._n_dofs], device=self._device),
            torch.zeros([self._batch_size, 3, self._n_dofs], device=self._device),
        )

        self._link_pose_dict:Dict[str, torch.Tensor] = {}
        for name in self._link_names:
            self._link_pose_dict[name] = (
                torch.zeros(self._batch_size, 3, device=self._device),
                torch.zeros(self._batch_size, 3, 3, device=self._device)
            )

        #load collision model
        self.collision_spheres = robot_config['collision_spheres']
        self.robot_collision_params = robot_config['robot_collision_params']


    def delete_lxml_objects(self):
        self._urdf_model = None
    
    def load_lxml_objects(self):
        self._urdf_model = URDFRobotModel(
            urdf_path=self.urdf_path, device=self._device) #, dtype=self.dtype
        # )

    def allocate_buffers(self, batch_size:int):
        pass

    # @tensor_check
    @torch.jit.export
    def update_kinematic_state(self, q: torch.Tensor, qd: torch.Tensor)->None:
        r"""

        Updates the kinematic state of the robot
        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]

        Returns:

        """
        # assert q.ndim == 2
        # assert qd.ndim == 2
        # assert q.shape[1] == self._n_dofs
        # assert qd.shape[1] == self._n_dofs

        batch_size = q.shape[0]

        if batch_size != self._batch_size:
            self._batch_size = batch_size
            self._base_lin_vel = torch.zeros((self._batch_size, 3), device=self._device) #, dtype=self.dtype)
            self._base_ang_vel = torch.zeros((self._batch_size, 3), device=self._device) #, dtype=self.dtype)
            self._base_pose_trans = torch.zeros((self._batch_size, 3), device=self._device) #, dtype=self.dtype)
            self._base_pose_rot = torch.eye(3, device=self._device) #, dtype=self.dtype).expand(self._batch_size,3,3)
            self._lin_jac, self._ang_jac = (
                torch.zeros([self._batch_size, 3, self._n_dofs], device=self._device),
                torch.zeros([self._batch_size, 3, self._n_dofs], device=self._device),
            )

        # update the state of the joints
        # for i in range(q.shape[1]):
        #     idx = self._controlled_joints[i]
        #     self._bodies[idx].update_joint_state(
        #         q[:, i].unsqueeze(1), qd[:, i].unsqueeze(1)
        #     )

        # we assume a non-moving base
        parent_body = self._bodies[0]
        parent_body.vel = SpatialMotionVec(
            self._base_lin_vel, self._base_ang_vel,
            device=self._device
        )
        # parent_body.vel = SpatialMotionVec(
        #     torch.zeros((batch_size, 3), device=self._device),
        #     torch.zeros((batch_size, 3), device=self._device),
        #     device=self._device
        # )

        # propagate the new joint state through the kinematic chain to update bodies position/velocities
        with record_function("robot_model:fk/for_loop"):

            # for i in range(1, len(self._bodies)):
            for i, (body, parent_body) in enumerate(zip(self._bodies[1:], self._bodies[0:-1]), 1):

                if i in self._body_idx_to_controlled_joint_idx_map:
                    idx = self._body_idx_to_controlled_joint_idx_map[i]
                    # self._bodies[i].update_joint_state(q[:,idx].unsqueeze(1), qd[:,idx].unsqueeze(1))
                    body.update_joint_state(q[:,idx].unsqueeze(1), qd[:,idx].unsqueeze(1))

                # body = self._bodies[i]
                # parent_name = self._urdf_model.get_name_of_parent_body(body.name)
                # find the joint that has this link as child
                # parent_body = self._bodies[self._name_to_idx_map[parent_name]]

                #### The steps marked with X correspond to calculation of link velocities in 
                # body frame. We are ommiting these for now, and they can be later included 
                # if link velocities are required
                #             
                # transformation operator from child link to parent link
                childToParentT = body.joint_pose
                
                # # X: transformation operator from parent link to child link
                # with record_function('robot_model:fk/for_loop/transform_inverse'):
                #     parentToChildT = childToParentT.inverse()

                # the position and orientation of the body in world coordinates, with origin at the joint
                with record_function('robot_model:fk/for_loop/multiply_transform'):
                    body.pose = parent_body.pose.multiply_transform(childToParentT)

                # # X: we rotate the velocity of the parent's body into the child frame
                # with record_function('robot_model:fk/for_loop/vel_transform'):
                #     new_vel = parent_body.vel.transform(parentToChildT)

                # # X this body's angular velocity is combination of the velocity experienced at it's parent's link
                # # + the velocity created by this body's joint
                # with record_function('robot_model:fk/for_loop/add_motion_vec'):
                #     body.vel = body.joint_vel.add_motion_vec(new_vel)

    # @tensor_check
    # @torch.jit.export
    # def compute_forward_kinematics_all_links(
    #     self, q: torch.Tensor
    # ) -> Dict[str, torch.Tensor]:
    #     r"""

    #     Args:
    #         q: joint angles [batch_size x n_dofs]
    #         link_name: name of link

    #     Returns: translation and rotation of the link frame

    #     """
    #     # Create joint state dictionary
    #     q_dict = {}
    #     for i, body_idx in enumerate(self._controlled_joints):
    #         q_dict[self._bodies[body_idx].name] = q[:, i].unsqueeze(1)

    #     # Call forward kinematics on root node
    #     pose_dict = self._bodies[0].forward_kinematics(q_dict)

    #     return {
    #         link: (pose_dict[link].translation(), pose_dict[link].rotation()) #.get_quaternion()
    #         for link in pose_dict.keys()
    #     }

    # @tensor_check
    @torch.jit.export
    def compute_forward_kinematics(
        self, q: torch.Tensor, qd:torch.Tensor #, link_name: str, 
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]: #recursive: bool = False
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            link_name: name of link

        Returns: translation and rotation of the link frame

        """
        # assert q.ndim == 2

        # if recursive:
        #     return self.compute_forward_kinematics_all_links(q)[link_name]

        # else:

        # qd = torch.zeros_like(q)
        with record_function('robot_model:update_kinematic_state'):
            self.update_kinematic_state(q, qd)

        link_pose_dict = self._link_pose_dict
        for link_name in self._link_names:
            pose = self._bodies[self._name_to_idx_map[link_name]].pose
            pos = pose.translation() #.to(inp_device)
            rot = pose.rotation() #.to(inp_device)#get_quaternion()        
            link_pose_dict[link_name] = (pos, rot)
        return link_pose_dict

    @torch.jit.export
    def get_link_pose(self, link_name: str)->Tuple[torch.Tensor, torch.Tensor]:
        pose = self._bodies[self._name_to_idx_map[link_name]].pose
        pos = pose.translation() #.to(inp_device)
        rot = pose.rotation() #.to(inp_device)#get_quaternion()
        return pos, rot

    # @tensor_check
    def iterative_newton_euler(self, base_acc: SpatialMotionVec) -> None:
        r"""

        Args:
            base_acc: spatial acceleration of base (for fixed manipulators this is zero)
        """

        body = self._bodies[0]
        body.acc = base_acc

        # forward pass to propagate accelerations from root to end-effector link
        for i in range(1, len(self._bodies)):
            body = self._bodies[i]
            parent_name = self._urdf_model.get_name_of_parent_body(body.name)

            parent_body = self._bodies[self._name_to_idx_map[parent_name]]

            # get the inverse of the current joint pose
            inv_pose = body.joint_pose.inverse()

            # transform spatial acceleration of parent body into this body's frame
            acc_parent_body = parent_body.acc.transform(inv_pose)
            # body velocity cross joint vel
            tmp = body.vel.cross_motion_vec(body.joint_vel)
            body.acc = acc_parent_body.add_motion_vec(body.joint_acc).add_motion_vec(
                tmp
            )

        # reset all forces for backward pass
        for i in range(0, len(self._bodies)):
            self._bodies[i].force = SpatialForceVec(device=self._device)

        # backward pass to propagate forces up (from endeffector to root body)
        for i in range(len(self._bodies) - 1, 0, -1):
            body = self._bodies[i]
            joint_pose = body.joint_pose

            # body force on joint
            icxacc = body.inertia.multiply_motion_vec(body.acc)
            icxvel = body.inertia.multiply_motion_vec(body.vel)
            tmp_force = body.vel.cross_force_vec(icxvel)

            body.force = body.force.add_force_vec(icxacc).add_force_vec(tmp_force)

            # pose x body_force => propagate to parent
            if i > 0:
                parent_name = self._urdf_model.get_name_of_parent_body(body.name)
                parent_body = self._bodies[self._name_to_idx_map[parent_name]]

                backprop_force = body.force.transform(joint_pose)
                parent_body.force = parent_body.force.add_force_vec(backprop_force)

        return

    # @tensor_check
    def compute_inverse_dynamics(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd_des: torch.Tensor,
        include_gravity: Optional[bool] = True,
        use_damping: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            qdd_des: desired joint accelerations [batch_size x n_dofs]
            include_gravity: when False, we assume gravity compensation is already taken care off

        Returns: forces to achieve desired accelerations

        """
        assert q.ndim == 2
        assert qd.ndim == 2
        assert qdd_des.ndim == 2
        assert q.shape[1] == self._n_dofs
        assert qd.shape[1] == self._n_dofs
        assert qdd_des.shape[1] == self._n_dofs

        batch_size = qdd_des.shape[0]
        force = torch.zeros_like(qdd_des)

        # we set the current state of the robot
        self.update_kinematic_state(q, qd)

        # we set the acceleration of all controlled joints to the desired accelerations
        for i in range(self._n_dofs):
            idx = self._controlled_joints[i]
            self._bodies[idx].update_joint_acc(qdd_des[:, i].unsqueeze(1))

        # forces at the base are either 0, or gravity
        base_ang_acc = q.new_zeros((batch_size, 3))
        base_lin_acc = q.new_zeros((batch_size, 3))
        if include_gravity:
            base_lin_acc[:, 2] = 9.81 * torch.ones(batch_size, device=self._device)

        # we propagate the base forces
        self.iterative_newton_euler(SpatialMotionVec(base_lin_acc, base_ang_acc))

        # we extract the relevant forces for all controlled joints
        for i in range(qdd_des.shape[1]):
            idx = self._controlled_joints[i]
            rot_axis = torch.zeros((batch_size, 3), device=self._device)
            axis = self._bodies[idx].joint_axis[0]
            axis_idx = int(torch.where(axis)[0])
            rot_sign = torch.sign(axis[axis_idx])

            rot_axis[:, axis_idx] = rot_sign * torch.ones(
                batch_size, device=self._device
            )
            force[:, i] += (
                self._bodies[idx].force.ang.unsqueeze(1) @ rot_axis.unsqueeze(2)
            ).squeeze()

        # we add forces to counteract damping
        if use_damping:
            damping_const = torch.zeros((1, self._n_dofs), device=self._device)
            for i in range(self._n_dofs):
                idx = self._controlled_joints[i]
                damping_const[:, i] = self._bodies[idx].get_joint_damping_const()
            force += damping_const.repeat(batch_size, 1) * qd

        return force

    # @tensor_check
    def compute_non_linear_effects(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        include_gravity: Optional[bool] = True,
        use_damping: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""

        Compute the non-linear effects (Coriolis, centrifugal, gravitational, and damping effects).

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns:

        """
        zero_qdd = q.new_zeros(q.shape)
        return self.compute_inverse_dynamics(
            q, qd, zero_qdd, include_gravity, use_damping
        )

    # @tensor_check
    def compute_lagrangian_inertia_matrix(
        self,
        q: torch.Tensor,
        include_gravity: Optional[bool] = True,
        use_damping: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns:

        """
        assert q.shape[1] == self._n_dofs
        batch_size = q.shape[0]
        identity_tensor = (
            torch.eye(q.shape[1], device=self._device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        zero_qd = q.new_zeros(q.shape)
        zero_qdd = q.new_zeros(q.shape)
        if include_gravity:
            gravity_term = self.compute_inverse_dynamics(
                q, zero_qd, zero_qdd, include_gravity, use_damping
            )
        else:
            gravity_term = q.new_zeros(q.shape)

        H = torch.stack(
            [
                (
                    self.compute_inverse_dynamics(
                        q,
                        zero_qd,
                        identity_tensor[:, :, j],
                        include_gravity,
                        use_damping,
                    )
                    - gravity_term
                )
                for j in range(self._n_dofs)
            ],
            dim=2,
        )
        return H

    # @tensor_check
    def compute_forward_dynamics_old(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        f: torch.Tensor,
        include_gravity: Optional[bool] = True,
        use_damping: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""
        Computes next qdd by solving the Euler-Lagrange equation
        qdd = H^{-1} (F - Cv - G - damping_term)

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            f: forces to be applied [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns: accelerations that are the result of applying forces f in state q, qd

        """

        nle = self.compute_non_linear_effects(
            q=q, qd=qd, include_gravity=include_gravity, use_damping=use_damping
        )
        inertia_mat = self.compute_lagrangian_inertia_matrix(
            q=q, include_gravity=include_gravity, use_damping=use_damping
        )

        # Solve H qdd = F - Cv - G - damping_term
        qdd = torch.solve(f.unsqueeze(2) - nle.unsqueeze(2), inertia_mat)[0].squeeze(2)

        return qdd

    # @tensor_check
    def compute_forward_dynamics(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        f: torch.Tensor,
        include_gravity: Optional[bool] = True,
        use_damping: Optional[bool] = False,
    ) -> torch.Tensor:
        r"""
        Computes next qdd via the articulated body algorithm (see Featherstones Rigid body dynamics page 132)

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            f: forces to be applied [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns: accelerations that are the result of applying forces f in state q, qd

        """
        assert q.ndim == 2
        assert qd.ndim == 2
        assert q.shape[1] == self._n_dofs
        assert qd.shape[1] == self._n_dofs

        qdd = torch.zeros_like(q)
        batch_size = q.shape[0]

        if use_damping:
            damping_const = torch.zeros((1, self._n_dofs), device=self._device)
            for i in range(self._n_dofs):
                idx = self._controlled_joints[i]
                damping_const[:, i] = self._bodies[idx].get_joint_damping_const()
            f -= damping_const.repeat(batch_size, 1) * qd

        # we set the current state of the robot
        self.update_kinematic_state(q, qd)

        # forces at the base are either 0, or gravity
        base_ang_acc = q.new_zeros((batch_size, 3))
        base_lin_acc = q.new_zeros((batch_size, 3))
        if include_gravity:
            base_lin_acc[:, 2] = 9.81 * torch.ones(batch_size, device=self._device)

        base_acc = SpatialMotionVec(base_lin_acc, base_ang_acc)

        body = self._bodies[0]
        body.acc = base_acc

        for i in range(1, len(self._bodies)):
            body = self._bodies[i]

            # body velocity cross joint vel
            body.c = body.vel.cross_motion_vec(body.joint_vel)
            icxvel = body.inertia.multiply_motion_vec(body.vel)
            body.pA = body.vel.cross_force_vec(icxvel)
            # IA is 6x6, we repeat it for each item in the batch, as the raw inertia matrix is shared across the whole batch
            body.IA = body.inertia.get_spatial_mat().repeat((batch_size, 1, 1))

        for i in range(len(self._bodies) - 1, 0, -1):
            body = self._bodies[i]

            S = SpatialMotionVec(
                lin_motion=torch.zeros((batch_size, 3), device=self._device),
                ang_motion=body.joint_axis.repeat((batch_size, 1)),
            )
            body.S = S
            Utmp = torch.bmm(body.IA, S.get_vector()[..., None])[..., 0]
            body.U = SpatialForceVec(lin_force=Utmp[:, 3:], ang_force=Utmp[:, :3])
            body.d = S.dot(body.U)
            if body.joint_idx is not None:
                body.u = f[:, body.joint_idx] - body.pA.dot(S)
            else:
                body.u = -body.pA.dot(S)

            parent_name = self._urdf_model.get_name_of_parent_body(body.name)
            parent_idx = self._name_to_idx_map[parent_name]

            if parent_idx > 0:
                parent_body = self._bodies[parent_idx]
                U = body.U.get_vector()
                Ud = U / (
                    body.d.view(batch_size, 1) + 1e-37
                )  # add smoothing values in case of zero mass
                c = body.c.get_vector()

                # IA is of size [batch_size x 6 x 6]
                IA = body.IA - torch.bmm(
                    U.view(batch_size, 6, 1), Ud.view(batch_size, 1, 6)
                )

                tmp = torch.bmm(IA, c.view(batch_size, 6, 1)).squeeze(dim=2)
                tmps = SpatialForceVec(lin_force=tmp[:, 3:], ang_force=tmp[:, :3])
                ud = body.u / (
                    body.d + 1e-37
                )  # add smoothing values in case of zero mass
                uu = body.U.multiply(ud)
                pa = body.pA.add_force_vec(tmps).add_force_vec(uu)

                joint_pose = body.joint_pose

                # transform is of shape 6x6
                transform_mat = joint_pose.to_matrix()
                if transform_mat.shape[0] != IA.shape[0]:
                    transform_mat = transform_mat.repeat(IA.shape[0], 1, 1)
                parent_body.IA += torch.bmm(transform_mat.transpose(-2, -1), IA).bmm(
                    transform_mat
                )
                parent_body.pA = parent_body.pA.add_force_vec(pa.transform(joint_pose))

        base_acc = SpatialMotionVec(lin_motion=base_lin_acc, ang_motion=base_ang_acc)

        body = self._bodies[0]
        body.acc = base_acc

        # forward pass to propagate accelerations from root to end-effector link
        for i in range(1, len(self._bodies)):
            body = self._bodies[i]
            parent_name = self._urdf_model.get_name_of_parent_body(body.name)
            parent_idx = self._name_to_idx_map[parent_name]
            parent_body = self._bodies[parent_idx]

            # get the inverse of the current joint pose
            inv_pose = body.joint_pose.inverse()

            # transform spatial acceleration of parent body into this body's frame
            acc_parent_body = parent_body.acc.transform(inv_pose)
            # body velocity cross joint vel
            body.acc = acc_parent_body.add_motion_vec(body.c)

            # Joint acc
            if i in self._controlled_joints:
                joint_idx = self._controlled_joints.index(i)
                qdd[:, joint_idx] = (1.0 / body.d) * (body.u - body.U.dot(body.acc))
                body.acc = body.acc.add_motion_vec(body.S.multiply(qdd[:, joint_idx]))

        return qdd

    # @tensor_check
    def compute_endeffector_jacobian(
        self, q: torch.Tensor, link_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            link_name: name of link name for the jacobian
            q: joint angles [batch_size x n_dofs]

        Returns: linear and angular jacobian

        """
        # assert len(q.shape) == 2
        batch_size = q.shape[0]
        self.compute_forward_kinematics(q, link_name)

        e_pose = self._bodies[self._name_to_idx_map[link_name]].pose
        p_e = e_pose.translation()

        # lin_jac, ang_jac = (
        #     torch.zeros([batch_size, 3, self._n_dofs], device=self._device),
        #     torch.zeros([batch_size, 3, self._n_dofs], device=self._device),
        # )
        lin_jac = self._lin_jac
        ang_jac = self._ang_jac

        joint_id = self._bodies[self._name_to_idx_map[link_name]].joint_id
        while link_name != self._bodies[0].name:
            if joint_id in self._controlled_joints:
                i = self._controlled_joints.index(joint_id)
                idx = joint_id

                pose = self._bodies[idx].pose
                axis = self._bodies[idx].joint_axis
                p_i = pose.translation()
                z_i = pose.rotation() @ axis.squeeze()
                lin_jac[:, :, i] = torch.cross(z_i, p_e - p_i, dim=-1)
                ang_jac[:, :, i] = z_i

            link_name = self._urdf_model.get_name_of_parent_body(link_name)
            joint_id = self._bodies[self._name_to_idx_map[link_name]].joint_id

        return lin_jac, ang_jac
    
    @torch.jit.export
    def compute_fk_and_jacobian(
            self, q:torch.Tensor, qd:torch.Tensor, link_name: str
    ) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:

        
        with record_function("robot_model:fk"):
            # ee_pos, ee_rot = self.compute_forward_kinematics(q, qd) #, link_name)
            link_pose_dict = self.compute_forward_kinematics(q, qd) #, link_name)

        ee_pos = link_pose_dict[link_name][0]
        ee_rot = link_pose_dict[link_name][1]

        # e_pose = self._bodies[self._name_to_idx_map[link_name]].pose
        # p_e = e_pose.translation()

        #use pre-allocated buffers
        lin_jac = self._lin_jac
        ang_jac = self._ang_jac

        parent_joint_id = self._bodies[self._name_to_idx_map[link_name]].joint_id
        
        with record_function("robot_model:jac"):
            # while link_name != self._bodies[0].name:
            for i, idx in enumerate(self._controlled_joints):
                # if joint_id in self._controlled_joints:
                #     i = self._controlled_joints.index(joint_id)
                #     idx = joint_id
                if (idx - 1) > parent_joint_id:
                    continue

                pose = self._bodies[idx].pose
                axis = self._bodies[idx].joint_axis
                p_i = pose.translation()
                z_i = pose.rotation() @ axis.squeeze()
                lin_jac[:, :, i] = torch.cross(z_i, ee_pos - p_i, dim=-1) #p_e
                ang_jac[:, :, i] = z_i

                # link_name = self._urdf_model.get_name_of_parent_body(link_name)
                # joint_id = self._bodies[self._name_to_idx_map[link_name]].joint_id

        # return ee_pos, ee_rot, lin_jac, ang_jac, ee_lin_vel, ee_ang_vel
        return link_pose_dict, lin_jac, ang_jac


    def _get_parent_object_of_param(self, link_name: str, parameter_name: str):
        body_idx = self._name_to_idx_map[link_name]
        if parameter_name in ["trans", "rot_angles", "joint_damping"]:
            parent_object = self._bodies[body_idx]
        elif parameter_name in ["mass", "inertia_mat", "com"]:
            parent_object = self._bodies[body_idx].inertia
        else:
            raise AttributeError(
                "Invalid parameter name. Accepted parameter names are: "
                "trans, rot_angles, joint_damping, mass, inertia_mat, com"
            )
        return parent_object

    def make_link_param_learnable(
        self, link_name: str, parameter_name: str, parametrization: torch.nn.Module
    ):
        parent_object = self._get_parent_object_of_param(link_name, parameter_name)

        # Replace current parameter with a learnable module
        parent_object.__delattr__(parameter_name)
        parent_object.add_module(parameter_name, parametrization.to(self._device))


    @torch.jit.export
    def get_joint_limits(self)-> List[Dict[str, float]]:  # -> List[Dict[str, torch.Tensor]]:
        r"""

        Returns: list of joint limit dict, containing joint position, velocity and effort limits

        """
        # limits = []
        # for idx in self._controlled_joints:
        #     limits.append(self._bodies[idx].get_joint_limits())
        # return limits
        return self._joint_limits

    @torch.jit.export
    def get_link_names(self) -> List[str]:
        r"""

        Returns: a list containing names for all links

        """

        return self._link_names

    def print_link_names(self) -> None:
        r"""

        print the names of all links

        """
        for i in range(len(self._bodies)):
            print(self._bodies[i].name)

if __name__ == "__main__":
    import os
    from storm_kit.util_file import get_assets_path
    import time 
    urdf_path = os.path.abspath(os.path.join(get_assets_path(), 'urdf/franka_description/franka_panda_no_gripper.urdf'))
    device = torch.device('cuda:0')
    robot_model = torch.jit.script(DifferentiableRobotModel(urdf_path, device=device))

    #generate fake data
    batch_size = 100000
    q_pos = torch.randn(batch_size, 7, device=device)
    q_vel = torch.randn(batch_size, 7, device=device)
    q_acc = torch.randn(batch_size, 7, device=device)
    
    st = time.time()
    robot_model.compute_fk_and_jacobian(q_pos, link_name='ee_link')
    print(time.time()-st)



# class DifferentiableKUKAiiwa(DifferentiableRobotModel):
#     def __init__(self, device=None):
#         rel_urdf_path = "kuka_iiwa/urdf/iiwa7.urdf"
#         self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
#         self.learnable_rigid_body_config = None
#         self.name = "differentiable_kuka_iiwa"
#         super().__init__(self.urdf_path, self.name, device=device)


# class DifferentiableFrankaPanda(DifferentiableRobotModel):
#     def __init__(self, device=None):
#         rel_urdf_path = "panda_description/urdf/panda_no_gripper.urdf"
#         self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
#         self.learnable_rigid_body_config = None
#         self.name = "differentiable_franka_panda"
#         super().__init__(self.urdf_path, self.name, device=device)


# class DifferentiableTwoLinkRobot(DifferentiableRobotModel):
#     def __init__(self, device=None):
#         rel_urdf_path = "2link_robot.urdf"
#         self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
#         self.learnable_rigid_body_config = None
#         self.name = "diff_2d_robot"
#         super().__init__(self.urdf_path, self.name, device=device)


# class DifferentiableTrifingerEdu(DifferentiableRobotModel):
#     def __init__(self, device=None):
#         rel_urdf_path = "trifinger_edu_description/trifinger_edu.urdf"
#         self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
#         self.learnable_rigid_body_config = None
#         self.name = "trifinger_edu"
#         super().__init__(self.urdf_path, self.name, device=device)
