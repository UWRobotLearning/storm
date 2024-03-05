#!/usr/bin/env python
"""
Node that publishes world and robot collision model.
"""
import os
import rospy
import rospkg
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

import torch
from hydra import initialize, compose

from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.spatial_vector_algebra import quaternion_to_matrix, CoordinateTransform
from storm_kit.geom.sdf.robot import RobotSphereCollision
from storm_kit.tasks.arm_reacher import ArmTask

class RobotWorldPublisher():
    def __init__(self) -> None:
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('storm_ros')
        self.storm_path = os.path.dirname(self.pkg_path)

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_names = rospy.get_param('~robot_joint_names', None)
        self.fixed_frame = rospy.get_param('~fixed_frame', None)
        
        # self.device = torch.device('cuda', 0)
        self.device = torch.device('cpu')

        initialize(config_path="../../../content/configs/gym", job_name="robot_world_publisher")
        self.config = compose(config_name="config", overrides=["task=FrankaReacherRealRobot"])
        self.robot_collision_config = self.config.task.robot.robot_collision_params
        self.ee_link_name = self.config.task.robot.ee_link_name
        self.n_dofs = self.config.task.joint_control.n_dofs
        self.urdf_path = join_path(get_assets_path(), self.config.task.robot.urdf_path)
        self.link_names = self.robot_collision_config['collision_link_names']
        self.world_params = self.config.task.world['world_model']
        self.task = ArmTask(
            cfg=self.config.task.task, world_cfg=self.config.task.world, device=self.device, viz_rollouts=False
        )
        self.tstep = 0
        self.tstep_tensor = torch.as_tensor([self.tstep], device=self.device).unsqueeze(0)
        self.start_t = None
        self.first_iter = True

        self.robot_state = {}
        self.robot_sphere_marker_array = MarkerArray()
        self.marker_pub = rospy.Publisher("/robot_collision_spheres", MarkerArray, queue_size=1, tcp_nodelay=True, latch=False)
        self.world_marker_pub = rospy.Publisher("/world", MarkerArray, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        self.rate = rospy.Rate(500)
        self.state_sub_on = False

        self.robot_model = torch.jit.script(DifferentiableRobotModel(self.config.task.robot,device=self.device))
        # self.robot_model = torch.jit.script(DifferentiableRobotModel(self.robot_cfg, device=self.device))
        self.collision_model = RobotSphereCollision(self.robot_collision_config)
        # self.collision_model.build_batch_features(batch_size=1, clone_pose=True, clone_objs=True)


    def robot_state_callback(self, msg):
        self.state_sub_on = True
        #save gripper state
        # self.gripper_state.header = msg.header
        # self.gripper_state.position = msg.position[0:2]
        # self.gripper_state.velocity = msg.velocity[0:2]
        # self.gripper_state.effort = msg.effort[0:2]

        # self.gripper_state['position'] = np.array(msg.position[0:2])
        # self.gripper_state['velocity'] = np.array(msg.velocity[0:2])

        # #save robot state
        # self.robot_state.header = msg.header
        # self.robot_state.position = msg.position[2:]
        # self.robot_state.velocity = msg.velocity[2:]
        # self.robot_state.effort = msg.effort[2:]
        self.robot_state['q_pos'] = torch.tensor(msg.position).unsqueeze(0)
        self.robot_state['q_vel'] = torch.tensor(msg.velocity).unsqueeze(0)
        self.robot_state['q_acc'] = torch.zeros_like(self.robot_state['q_vel'])

    def pub_loop(self):
        while not rospy.is_shutdown():
            if self.state_sub_on:
                self.tstep_tensor[0,0] = self.tstep
                self.robot_state['tstep'] = self.tstep_tensor

                #publish robot spheres
                robot_spheres = self.get_robot_collision_spheres(self.robot_state)
                marker_list = self.get_sphere_marker_list(robot_spheres)
                self.robot_sphere_marker_array.markers = marker_list

                _, cost, done_task, cost_terms, done_cost, done_info = self.task.forward(self.robot_state, None)
                # print(cost_terms['manip_cost'], cost_terms['manip_score'])
                # print(done_info['self_coll_dist'], done_info['self_cost_w'])
                #publish world
                self.marker_pub.publish(self.robot_sphere_marker_array)
                world_marker_list = self.get_world_marker_list()
                self.world_marker_pub.publish(world_marker_list)

            self.rate.sleep()

    def get_robot_collision_spheres(self, robot_state):
        link_pose_dict = self.robot_model.compute_forward_kinematics(
            robot_state['q_pos'], robot_state['q_vel']) # link_name=self.ee_link_name)
        
        # link_pos_seq, link_rot_seq = [], []
        # for _,k in enumerate(self.link_names):
        #     # link_pos, link_rot = self.robot_model.get_link_pose(k)
        #     link_pos, link_rot = link_pose_dict[k]
        #     link_pos_seq.append(link_pos.unsqueeze(1))
        #     link_rot_seq.append(link_rot.unsqueeze(1))

        # link_pos_seq = torch.cat(link_pos_seq, axis=1)
        # link_rot_seq = torch.cat(link_rot_seq, axis=1)
        # self.collision_model.update_batch_robot_collision_objs(link_pos_seq, link_rot_seq)
        # print(link_pose_dict[0])
        for link_name in self.link_names:
            link_pos  = link_pose_dict[0][link_name][0].unsqueeze(1)
            link_rot = link_pose_dict[0][link_name][1].unsqueeze(1)
            link_pose_dict[0][link_name] = (link_pos, link_rot)

        self.collision_model.update_batch_robot_collision_objs(link_pose_dict)
        spheres = self.collision_model.get_batch_robot_link_spheres()
        spheres_list = []
        for k in spheres:
            spheres_list.append(spheres[k].numpy())


        # self.collision_model.update_batch_robot_collision_objs(link_pos_seq, link_rot_seq)
        # res = self.collision_model.check_self_collisions(link_pos_seq, link_rot_seq)
        # res = torch.max(res, dim=-1)[0]
        return spheres_list

    def get_sphere_marker_list(self, sphere_list):
        markers_list = []
        for i, link_spheres in enumerate(sphere_list):
            #each link has multiple spheres
            spheres = link_spheres[0]
            for j in range(spheres.shape[0]):
                link_sp = spheres[j] 
                x, y, z, r = link_sp
                sphere_marker = Marker()
                sphere_marker.header.frame_id = self.fixed_frame
                sphere_marker.ns = self.link_names[i] + "/sphere{}".format(j) # unique ID
                sphere_marker.type = Marker().SPHERE
                sphere_marker.action = Marker().ADD
                # sphere_marker.lifetime = self.marker_lifetime
                sphere_marker.pose.position.x = x
                sphere_marker.pose.position.y = y
                sphere_marker.pose.position.z = z
                sphere_marker.pose.orientation.x = 0.0
                sphere_marker.pose.orientation.y = 0.0
                sphere_marker.pose.orientation.z = 0.0
                sphere_marker.pose.orientation.w = 1.0
                sphere_marker.scale.x = 2.0*r
                sphere_marker.scale.y = 2.0*r
                sphere_marker.scale.z = 2.0*r
                sphere_marker.color.a = 0.5
                sphere_marker.color.r = 1.0
                sphere_marker.color.g = 0.0
                sphere_marker.color.b = 0.0
                # point1 = Point()
                # self.sphere_marker2.points.append(point1)
                # self.sphere_marker2.colors.append(self.getColor('blue'))
                markers_list.append(sphere_marker)
        
        return markers_list


    def get_world_marker_list(self):
        markers_list = []
        spheres_objs = self.world_params['coll_objs']['sphere']
        cube_objs = self.world_params['coll_objs']['cube']

        #get sphere markers
        for k,v in spheres_objs.items():
            r = v['radius']
            x,y,z = v['position']
            sphere_marker = Marker()
            sphere_marker.header.frame_id = self.fixed_frame
            sphere_marker.ns =  k # unique ID
            sphere_marker.type = Marker().SPHERE
            sphere_marker.action = Marker().ADD
            # sphere_marker.lifetime = self.marker_lifetime
            sphere_marker.pose.position.x = x
            sphere_marker.pose.position.y = y
            sphere_marker.pose.position.z = z
            sphere_marker.pose.orientation.x = 0.0
            sphere_marker.pose.orientation.y = 0.0
            sphere_marker.pose.orientation.z = 0.0
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.scale.x = r
            sphere_marker.scale.y = r
            sphere_marker.scale.z = r
            sphere_marker.color.a = 0.8
            sphere_marker.color.r = 0.75
            sphere_marker.color.g = 0.75
            sphere_marker.color.b = 0.75
            # point1 = Point()
            # self.sphere_marker2.points.append(point1)
            # self.sphere_marker2.colors.append(self.getColor('blue'))
            markers_list.append(sphere_marker)


        #get cube markers
        for k,v in cube_objs.items():
            x_dim, y_dim, z_dim = v['dims']
            x, y, z, qx, qy, qz, qw = v['pose']
            cube_marker = Marker()
            cube_marker.header.frame_id = self.fixed_frame
            cube_marker.ns =  k # unique ID
            cube_marker.type = Marker().CUBE
            cube_marker.action = Marker().ADD
            # cube_marker.lifetime = self.marker_lifetime
            cube_marker.pose.position.x = x
            cube_marker.pose.position.y = y
            cube_marker.pose.position.z = z
            cube_marker.pose.orientation.x = qx
            cube_marker.pose.orientation.y = qy
            cube_marker.pose.orientation.z = qz
            cube_marker.pose.orientation.w = qw
            cube_marker.scale.x = x_dim
            cube_marker.scale.y = y_dim
            cube_marker.scale.z = z_dim
            cube_marker.color.a = 0.8
            cube_marker.color.r = 0.75
            cube_marker.color.g = 0.75
            cube_marker.color.b = 0.75
            # point1 = Point()
            # self.cube_marker2.points.append(point1)
            # self.cube_marker2.colors.append(self.getColor('blue'))
            markers_list.append(cube_marker)        


        return markers_list

    def close(self):
        self.state_sub.unregister()

if __name__ == "__main__":
    rospy.init_node("robot_world_publisher", anonymous=True, disable_signals=True)    
    torch.set_default_dtype(torch.float32)

    obj = RobotWorldPublisher()

    try:
        obj.pub_loop()
    except KeyboardInterrupt:
        obj.close()