# Copyright (c) Facebook, Inc. and its affiliates.
"""
Differentiable robot model class
====================================
TODO
"""

import copy
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os

import torch

from storm_kit.differentiable_robot_model.rigid_body import (
    DifferentiableRigidBody,
)
from storm_kit.differentiable_robot_model.spatial_vector_algebra import SpatialMotionVec, SpatialForceVec, transform_point
from storm_kit.differentiable_robot_model.urdf_utils import URDFRobotModel
from storm_kit.geom.geom_types import tensor_sphere
from storm_kit.util_file import join_path, get_assets_path
from torch.profiler import record_function
from hydra import initialize, compose


# import diff_robot_data

# robot_description_folder = diff_robot_data.__path__[0]


# def tensor_check(function):
#     """
#     A decorator for checking the device of input tensors
#     """

#     @dataclass
#     class BatchInfo:
#         shape: torch.Size = torch.Size([])
#         init: bool = False

#     def preprocess(arg, obj, batch_info):
#         if type(arg) is torch.Tensor:
#             # Check device
#             assert (
#                 arg.device.type == obj._device.type
#             ), f"Input argument of different device as module: {arg}"

#             # Check dimensions & convert to 2-dim tensors
#             assert arg.ndim in [1, 2], f"Input tensors must have ndim of 1 or 2."

#             if batch_info.init:
#                 assert (
#                     batch_info.shape == arg.shape[:-1]
#                 ), "Batch size mismatch between input tensors."
#             else:
#                 batch_info.init = True
#                 batch_info.shape = arg.shape[:-1]

#             if len(batch_info.shape) == 0:
#                 return arg.unsqueeze(0)

#         return arg

#     def postprocess(arg, batch_info):
#         if type(arg) is torch.Tensor and batch_info.init and len(batch_info.shape) == 0:
#             return arg[0, ...]

#         return arg

#     def wrapper(self, *args, **kwargs):
#         batch_info = BatchInfo()

#         # Parse input
#         processed_args = [preprocess(arg, self, batch_info) for arg in args]
#         processed_kwargs = {
#             key: preprocess(kwargs[key], self, batch_info) for key in kwargs
#         }

#         # Perform function
#         ret = function(self, *processed_args, **processed_kwargs)

#         # Parse output
#         if type(ret) is torch.Tensor:
#             return postprocess(ret, batch_info)
#         elif type(ret) is tuple:
#             return tuple([postprocess(r, batch_info) for r in ret])
#         else:
#             return ret

#     return wrapper


class DifferentiableRobotModel(torch.nn.Module):
    """
    Differentiable Robot Model
    ====================================
    """
    _controlled_joints: List[int]
    _n_dofs: int
    _n_links: int
    _name_to_idx_map: Dict[str, int]
    _body_idx_to_controlled_joint_idx_map: Dict[int, int]
    _name_to_parent_map: Dict[str, str]
    _body_to_joint_idx_map: Dict[str, int]
    _bodies: List[DifferentiableRigidBody]
    _joint_limits: List[Dict[str, float]]
    _link_names: List[str]
    _link_pose_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    _collision_link_names: List[str]
    _self_collision_ignore_dict: Dict[str, List[str]]

    # def __init__(self, urdf_path:str, name:str="", device:torch.device=torch.device('cpu')):
    def __init__(self, robot_config, device:torch.device=torch.device('cpu')):

        super().__init__()

        self.config = robot_config
        self.name = self.config['name']
        self._device = device
        self.urdf:str = join_path(get_assets_path(), self.config['urdf_path'])
        self.mesh_dir:str = join_path(get_assets_path(), self.config['mesh_dir'])
        self._urdf_model = URDFRobotModel(urdf_path=self.urdf, device=self._device)
        self._bodies = [] #torch.nn.ModuleList()
        self._n_dofs:int = 0
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
            body = torch.jit.script(DifferentiableRigidBody(
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
        self._n_links = len(self._bodies)
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

        self.q_pos = torch.zeros([self._batch_size, self._n_dofs], device=self._device)
        self.q_vel = torch.zeros([self._batch_size, self._n_dofs], device=self._device)
        self._lin_jac, self._ang_jac = (
            torch.zeros([self._batch_size, 3, self._n_dofs], device=self._device),
            torch.zeros([self._batch_size, 3, self._n_dofs], device=self._device),
        )

        self._link_pose_dict: Dict[str, Tuple[torch.Tensor]] = dict()
        for name in self._link_names:
            self._link_pose_dict[name] = (
                torch.zeros(self._batch_size, 3, device=self._device),
                torch.zeros(self._batch_size, 3, 3, device=self._device)
            )


        #Loading the collision model
        self._robot_collision_params = self.config.robot_collision_params
        self._collision_link_names: List[str] = self._robot_collision_params['collision_link_names']
        self._self_collision_ignore_dict: Dict[str, List[str]] = self._robot_collision_params['self_collision_ignore']
        self._all_collision_spheres = self._robot_collision_params['collision_spheres']
        self._link_collision_spheres: Dict[str, torch.Tensor] = dict()
        self._batch_link_collision_spheres: Dict[str, torch.Tensor] = dict()
        self._w_batch_link_collision_spheres: Dict[str, torch.Tensor] = dict() #this dict will holds link sphere locations in the world frame
        self._collision_link_to_idx: Dict[str, int] = dict()
        self._collision_idx_to_link:Dict[int, str] = dict()
        
        for (link_idx, link_name) in enumerate(self._collision_link_names):
            #Get number of spheres per link
            n_spheres = len(self._all_collision_spheres[link_name])
            link_spheres_tensor = torch.zeros((n_spheres, 4), device=self._device)

            for i in range(n_spheres):
                link_spheres_tensor[i,:] = tensor_sphere(
                    self._all_collision_spheres[link_name][i]['center'], self._all_collision_spheres[link_name][i]['radius'], 
                    device=self._device, tensor=link_spheres_tensor[i,:])
            self._link_collision_spheres[link_name] = link_spheres_tensor
            self._batch_link_collision_spheres[link_name] = link_spheres_tensor.unsqueeze(0).repeat(self._batch_size, 1, 1).clone()
            self._w_batch_link_collision_spheres[link_name] = link_spheres_tensor.unsqueeze(0).repeat(self._batch_size, 1, 1).clone()
            self._collision_link_to_idx[link_name] = link_idx
            self._collision_idx_to_link[link_idx] = link_name

        self._num_collision_links:int = len(self._w_batch_link_collision_spheres.keys())
        self._self_collision_dist_buff = torch.zeros(
            (self._batch_size, self._num_collision_links, self._num_collision_links), device=self._device)
        # self._self_collision_dist_buff2 = torch.zeros(
        #     (self._batch_size, self._num_collision_links, self._num_collision_links), device=self._device)


    def delete_lxml_objects(self):
        self._urdf_model = None
    
    def load_lxml_objects(self):
        self._urdf_model = URDFRobotModel(
            urdf_path=self._urdf_path, device=self._device)
        
    def allocate_buffers(self, batch_size:int):
        self._base_lin_vel = torch.zeros((batch_size, 3), device=self._device)
        self._base_ang_vel = torch.zeros((batch_size, 3), device=self._device)
        self._base_pose_trans = torch.zeros((batch_size, 3), device=self._device)
        self._base_pose_rot = torch.eye(3, device=self._device)
        self._lin_jac, self._ang_jac = (
            torch.zeros([batch_size, 3, self._n_dofs], device=self._device),
            torch.zeros([batch_size, 3, self._n_dofs], device=self._device),
        )

        for name in self._link_names:
            self._link_pose_dict[name] = (
                torch.zeros(batch_size, 3, device=self._device),
                torch.zeros(batch_size, 3, 3, device=self._device)
            )

        for k in self._batch_link_collision_spheres.keys():
            self._batch_link_collision_spheres[k] = self._link_collision_spheres[k].unsqueeze(0).repeat(batch_size, 1, 1).clone()
            self._w_batch_link_collision_spheres[k] = self._link_collision_spheres[k].unsqueeze(0).repeat(batch_size, 1, 1).clone()


        self._self_collision_dist_buff = torch.zeros(
            (batch_size, self._num_collision_links, self._num_collision_links), device=self._device)
        # self._self_collision_dist_buff2 = torch.zeros(
        #     (batch_size, self._num_collision_links, self._num_collision_links), device=self._device)

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
        self.q_pos = q
        self.q_vel = qd

        if batch_size != self._batch_size:
            self._batch_size = batch_size
            self.allocate_buffers(batch_size)

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
                link_name = body.name
                link_pos = body.pose.translation()
                link_rot = body.pose.rotation()
                self._link_pose_dict[link_name] = (link_pos, link_rot)
                
                if link_name in self._collision_link_names:
                    self._w_batch_link_collision_spheres[link_name][...,:3] = transform_point(
                        self._batch_link_collision_spheres[link_name][...,:3], link_rot, link_pos.unsqueeze(-2)
                    )

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
    ) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor], torch.Tensor]: #recursive: bool = False
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
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

        dist = find_link_distance(self._w_batch_link_collision_spheres, self._self_collision_ignore_dict, self._collision_idx_to_link, self._self_collision_dist_buff)
        # dist2 = find_link_distance2(self._w_batch_link_collision_spheres, self._self_collision_ignore_dict, self._collision_idx_to_link, self._self_collision_dist_buff2)
        
        # assert torch.allclose(dist, dist2)
        
        
        return self._link_pose_dict, self._w_batch_link_collision_spheres, dist
    
        # link_pose_dict = self._link_pose_dict
        # for link_name in self._link_names:
        #     pose = self._bodies[self._name_to_idx_map[link_name]].pose
        #     pos = pose.translation() #.to(inp_device)
        #     rot = pose.rotation() #.to(inp_device)#get_quaternion()        
        #     link_pose_dict[link_name] = (pos, rot)
        # return link_pose_dict

        # body = self._bodies[self._name_to_idx_map[link_name]]
        # pose = body.pose
        # vel = body.vel
        # pos = pose.translation()
        # rot = pose.rotation() #.get_quaternion()
        # return pos, rot

    @torch.jit.export
    def compute_fk_and_jacobian(
            self, q:torch.Tensor, qd:torch.Tensor, link_name: str
    ) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:

        # st = time.time()
        with record_function("robot_model:fk"):
            # ee_pos, ee_rot = self.compute_forward_kinematics(q, qd) #, link_name)
            link_pose_dict, link_collision_spheres_dict, self_collision_dist = self.compute_forward_kinematics(q, qd) #, link_name)
        ee_pos = link_pose_dict[link_name][0]
        ee_rot = link_pose_dict[link_name][1]
        # print('Time taken for FK:', time.time()-st)
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
        return link_pose_dict, link_collision_spheres_dict, self_collision_dist, lin_jac, ang_jac

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

    @property
    def n_dofs(self)->int:
        return self._n_dofs

    @property
    def n_links(self)->int:
        return self._n_links

    @property
    def n_collision_links(self)->int:
        return self._num_collision_links

    # def _get_parent_object_of_param(self, link_name: str, parameter_name: str):
    #     body_idx = self._name_to_idx_map[link_name]
    #     if parameter_name in ["trans", "rot_angles", "joint_damping"]:
    #         parent_object = self._bodies[body_idx]
    #     elif parameter_name in ["mass", "inertia_mat", "com"]:
    #         parent_object = self._bodies[body_idx].inertia
    #     else:
    #         raise AttributeError(
    #             "Invalid parameter name. Accepted parameter names are: "
    #             "trans, rot_angles, joint_damping, mass, inertia_mat, com"
    #         )
    #     return parent_object

    # def make_link_param_learnable(
    #     self, link_name: str, parameter_name: str, parametrization: torch.nn.Module
    # ):
    #     parent_object = self._get_parent_object_of_param(link_name, parameter_name)

    #     # Replace current parameter with a learnable module
    #     parent_object.__delattr__(parameter_name)
    #     parent_object.add_module(parameter_name, parametrization.to(self._device))

    # def freeze_learnable_link_param(self, link_name: str, parameter_name: str):
    #     parent_object = self._get_parent_object_of_param(link_name, parameter_name)

    #     # Get output value of current module
    #     param_module = getattr(parent_object, parameter_name)
    #     assert (
    #         type(param_module).__bases__[0] is torch.nn.Module
    #     ), f"{parameter_name} of {link_name} is not a learnable module."

    #     for param in param_module.parameters():
    #         param.requires_grad = False

    # def unfreeze_learnable_link_param(self, link_name: str, parameter_name: str):
    #     parent_object = self._get_parent_object_of_param(link_name, parameter_name)

    #     # Get output value of current module
    #     param_module = getattr(parent_object, parameter_name)
    #     assert (
    #         type(param_module).__bases__[0] is torch.nn.Module
    #     ), f"{parameter_name} of {link_name} is not a learnable module."

    #     for param in param_module.parameters():
    #         param.requires_grad = True

    # def print_link_names(self) -> None:
    #     r"""

    #     print the names of all links

    #     """
    #     for i in range(len(self._bodies)):
    #         print(self._bodies[i].name)

    # def print_learnable_params(self) -> None:
    #     r"""

    #     print the name and value of all learnable parameters

    #     """
    #     for name, param in self.named_parameters():
    #         print(f"{name}: {param}")
    

@torch.jit.script
def compute_spheres_distance(spheres_1:torch.Tensor, spheres_2:torch.Tensor)->torch.Tensor:
    b, n, _ = spheres_1.shape
    
    j = 0
    link_sphere_pts = spheres_1[:,j,:]
    link_sphere_pts = link_sphere_pts.unsqueeze(1)
    # find closest distance to other link spheres:
    s_dist = torch.norm(spheres_2[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
    s_dist = spheres_2[:,:,3] + link_sphere_pts[:,:,3] - s_dist
    max_dist = torch.max(s_dist, dim=-1)[0]
    
    for j in range(1, n):
        link_sphere_pts = spheres_1[:,j,:]
        link_sphere_pts = link_sphere_pts.unsqueeze(1)
        # find closest distance to other link spheres:
        s_dist = torch.norm(spheres_2[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
        s_dist = spheres_2[:,:,3] + link_sphere_pts[:,:,3] - s_dist
        s_dist = torch.max(s_dist, dim=-1)[0]
        max_dist = torch.maximum(max_dist, s_dist)
        
    dist = max_dist #torch.max(dist,dim=-1)[0]
    return dist

@torch.jit.script
def compute_spheres_distance2(spheres_1:torch.Tensor, spheres_2:torch.Tensor)->torch.Tensor:

    batch_sdist =  torch.cdist(spheres_1[...,0:3], spheres_2[...,0:3], p=2.0)
    spheres_2_radius = spheres_2[...,3]
    spheres_1_radius = spheres_1[...,3]
    batch_sdist = spheres_1_radius[...,None] + spheres_2_radius[:,None,:] - batch_sdist
    max_dist = torch.amax(batch_sdist, dim=(-1,-2)) #closest signed distance between current and query spheres
    return max_dist



@torch.jit.script
def find_link_distance(links_sphere_dict: Dict[str, torch.Tensor], collision_ignore_dict:Dict[str, List[str]], idx2link:Dict[int, str], dist: torch.Tensor)->torch.Tensor:
    futures : List[torch.jit.Future[torch.Tensor]] = []

    link_names = links_sphere_dict.keys()
    n_links = len(link_names)
    dist *= 0.0
    dist -= 100.0


    for link1_idx in range(n_links):
        # for every link, compute the distance to the other links:
        link1_name = idx2link[link1_idx]
        current_spheres = links_sphere_dict[link1_name]

        for link2_idx in range(link1_idx + 1, n_links):
            link2_name = idx2link[link2_idx]
            if link2_name not in collision_ignore_dict[link1_name]:
                compute_spheres = links_sphere_dict[link2_name]
                # find the distance between the two links:
                d = torch.jit.fork(compute_spheres_distance, current_spheres, compute_spheres)
                futures.append(d)

    #Gather results
    k = 0
    for link1_idx in range(n_links):
        link1_name = idx2link[link1_idx]
        for link2_idx in range(link1_idx + 1, n_links):
            link2_name = idx2link[link2_idx]
            if link2_name not in collision_ignore_dict[link1_name]:
                d = torch.jit.wait(futures[k])
                dist[:,link1_idx,link2_idx] = d
                dist[:,link2_idx,link1_idx] = d
                k += 1
    
    link_dist = torch.max(dist, dim=-1)[0]
    
    return link_dist


# @torch.jit.script
# def find_link_distance2(links_sphere_dict: Dict[str, torch.Tensor], collision_ignore_dict:Dict[str, List[str]], idx2link:Dict[int, str], dist: torch.Tensor)->torch.Tensor:

#     link_names = links_sphere_dict.keys()
#     n_links = len(link_names)
#     dist *= 0.0
#     dist -= 100.0

#     for link1_idx in range(n_links):
#         # for every link, compute the distance to the other links:
#         link1_name = idx2link[link1_idx]
#         current_spheres = links_sphere_dict[link1_name]

#         for link2_idx in range(link1_idx + 1, n_links):
#             link2_name = idx2link[link2_idx]
#             if link2_name not in collision_ignore_dict[link1_name]:
#                 query_spheres = links_sphere_dict[link2_name]
#                 batch_sdist =  torch.cdist(current_spheres[...,0:3], query_spheres[...,0:3], p=2.0)
#                 query_spheres_radius = query_spheres[...,3]
#                 current_spheres_radius = current_spheres[...,3]
#                 batch_sdist = current_spheres_radius[...,None] + query_spheres_radius[:,None,:] - batch_sdist
                
#                 max_dist = torch.amax(batch_sdist, dim=(-1,-2)) #closest signed distance between current and query spheres
#                 dist[:,link1_idx,link2_idx] = max_dist
#                 dist[:,link2_idx,link1_idx] = max_dist
    
#     link_closest_dist = torch.max(dist, dim=-1)[0]
    
#     return link_closest_dist


if __name__ == "__main__":
    import os
    from storm_kit.util_file import get_assets_path
    import time 
    # urdf_path = os.path.abspath(os.path.join(get_assets_path(), 'urdf/franka_description/franka_panda_no_gripper.urdf'))
    initialize(config_path="../../content/configs/gym", job_name="test")
    config = compose(config_name="config", overrides=["task=FrankaReacherRealRobot"])
    device = torch.device('cuda:0')
    robot_model = torch.jit.script(DifferentiableRobotModel(config.task.robot, device=device))

    #generate fake data
    batch_size = 12000
    q_pos = torch.randn(batch_size, 7, device=device)
    q_vel = torch.randn(batch_size, 7, device=device)
    q_acc = torch.randn(batch_size, 7, device=device)
    
    st = time.time()
    # robot_model.compute_fk_and_jacobian(q_pos, link_name='ee_link')
    link_pose_dict, link_spheres_dict, self_coll_dist, lin_jac, ang_jac = robot_model.compute_fk_and_jacobian(
                q_pos, q_vel,
                link_name='ee_link')
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
