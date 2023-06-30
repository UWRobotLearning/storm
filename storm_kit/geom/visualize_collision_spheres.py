#!/usr/bin/env python
import torch
import hydra
from omegaconf import DictConfig

from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path

from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.geom.sdf.robot import RobotSphereCollision

from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.plot_utils import plot_sphere
import matplotlib.pyplot as plt

@hydra.main(config_name="config", config_path= get_configs_path() + "/gym")
def visualize_collision_spheres(cfg: DictConfig):
    rollout_config = cfg.task.rollout
    robot_collision_config = rollout_config.model.robot_collision_params
    ee_link_name = rollout_config.model.ee_link_name
    
    urdf_path = join_path(get_assets_path(), robot_collision_config['urdf_path'])
    link_names = robot_collision_config['link_names']

    robot_model = DifferentiableRobotModel(urdf_path)
    collision_model = RobotSphereCollision(robot_collision_config)
    collision_model.build_batch_features(batch_size=1, clone_pose=True, clone_objs=True)


    q_pos = torch.tensor([0.0, -0.7853, 0.0, -2.3561, 0.0, 1.5707, 0.7853]).unsqueeze(0)
    q_vel = torch.zeros_like(q_pos)

    _,_,_,_ = robot_model.compute_fk_and_jacobian(
        q_pos, q_vel, link_name= ee_link_name)
    link_pos_seq, link_rot_seq = [], []

    for ki,k in enumerate(link_names):
        link_pos, link_rot = robot_model.get_link_pose(k)
        link_pos_seq.append(link_pos.unsqueeze(1))
        link_rot_seq.append(link_rot.unsqueeze(1))

    link_pos_seq = torch.cat(link_pos_seq, axis=1)
    link_rot_seq = torch.cat(link_rot_seq, axis=1)
    collision_model.update_batch_robot_collision_objs(link_pos_seq, link_rot_seq)
    # print('here')
    spheres = collision_model.get_batch_robot_link_spheres()
    spheres = [s.numpy() for s in spheres] 

    #visualize robot urdf with spheres
    tm = UrdfTransformManager()
    mesh = 'urdf/franka_description'
    mesh_path = join_path(get_assets_path(), mesh)
    with open(urdf_path, "r") as f:
        tm.load_urdf(f.read(), mesh_path=mesh_path)
    joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    for i,j in enumerate(joint_names):
        tm.set_joint(j, q_pos[0][i])
    ax = tm.plot_frames_in(
        "base_link", s=0.1,
        show_name=False)
    # ax = tm.plot_connections_in("lower_cone", ax=ax)
    tm.plot_visuals("base_link", ax=ax, convex_hull_of_mesh=True)

    for sphere in spheres:
        link_spheres = sphere[0]
        for sp in link_spheres:
            x, y, z, r = sp
            plot_sphere(ax, r, [x,y,z], wireframe=False, alpha=0.3, color='g')

    plt.show()


if __name__ == "__main__":
    visualize_collision_spheres()