import torch
import numpy as np
import yaml
import copy
import hydra
from omegaconf import DictConfig

from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask

from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.plot_utils import *
import matplotlib.pyplot as plt

def visualize_collision_spheres(cfg: DictConfig):
    robot_file = 'franka.yml'
    task_file = 'franka_real_robot_tray_reacher.yml'
    world_file = 'collision_wall_of_boxes.yml'
    world_yml = join_path(get_gym_configs_path(), world_file)

    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_yml = join_path(get_gym_configs_path(), robot_file)
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)

    tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)

    #set a dummy start state and goal
    g_pos = np.array([0.1, 0.1, 0.1])
    g_q = np.array([1.0, 0.0, 0.0, 0.0])
    mpc_control.update_params(goal_ee_pos=g_pos,
                                goal_ee_quat=g_q)
    current_robot_state = {
        'position': np.array([0.0, -0.7853, 0.0, -2.3561, 0.0, 1.5707, 0.7853]),
        'velocity': np.zeros(7),
        'acceleration': np.zeros(7)
    }

    # command = mpc_control.get_command(0.0, current_robot_state, control_dt=0.02, WAIT=True)
    current_robot_state_arr = np.hstack((current_robot_state['position'], current_robot_state['velocity'], current_robot_state['acceleration']))
    current_robot_state_arr = torch.as_tensor(current_robot_state_arr, **tensor_args).unsqueeze(0)
    current_cost = mpc_control.controller.rollout_fn.current_cost(current_robot_state_arr)

    link_pos_seq = copy.deepcopy(mpc_control.controller.rollout_fn.link_pos_seq)
    link_rot_seq = copy.deepcopy(mpc_control.controller.rollout_fn.link_rot_seq)
    batch_size = link_pos_seq.shape[0]
    horizon = link_pos_seq.shape[1]
    n_links = link_pos_seq.shape[2]
    link_pos = link_pos_seq.view(batch_size * horizon, n_links, 3)
    link_rot = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)

    mpc_control.controller.rollout_fn.robot_self_collision_cost.coll.update_batch_robot_collision_objs(link_pos, link_rot)

    spheres = mpc_control.controller.rollout_fn.robot_self_collision_cost.coll.w_batch_link_spheres
    spheres = [s.numpy() for s in spheres] 

    #visualize robot urdf with spheres
    tm = UrdfTransformManager()
    task_file = join_path(mpc_configs_path(), task_file)
    with open(task_file) as file:
        task_params = yaml.load(file, Loader=yaml.FullLoader)
    urdf = task_params['model']['urdf_path']
    mesh = 'urdf/franka_description/'
    urdf = join_path(get_assets_path(), urdf)
    mesh = join_path(get_assets_path(), mesh)
    print(urdf, mesh)
    with open(urdf, "r") as f:
        tm.load_urdf(f.read(), mesh_path=mesh)
    joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    for i,j in enumerate(joint_names):
        tm.set_joint(j, current_robot_state['position'][i])
    ax = tm.plot_frames_in(
        "base_link", s=0.1,
        show_name=False)
    # ax = tm.plot_connections_in("lower_cone", ax=ax)
    tm.plot_visuals("base_link", ax=ax, convex_hull_of_mesh=True)

    for sphere in spheres:
    # sphere = spheres[-1]
        # print(sphere)
        # exit()
        link_spheres = sphere[0]
        for sp in link_spheres:
            x, y, z, r = sp
            plot_sphere(ax, r, [x,y,z], wireframe=False, alpha=0.3, color='g')


    plt.show()


if __name__ == "__main__":
    visualize_collision_spheres()