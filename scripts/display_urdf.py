import os
import sys
import argparse

from storm_kit.util_file import get_urdf_path
import urdfpy

urdf_path = get_urdf_path()

parser = argparse.ArgumentParser(usage='Load an URDF file')
parser.add_argument('file', type=str, nargs='?',
                    default='franka_description/franka_panda_no_gripper.urdf', help='File to load. Use - for stdin')
parser.add_argument('-a', action='store_true',
                    help='Visualize robot articulation')
parser.add_argument('-c', action='store_true',
                    help='Use collision geometry')

args = parser.parse_args()

robot_default_dof_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.7853,
    "panda_joint3": 0.0,
    "panda_joint4": -2.3561,
    "panda_joint5": 0.0,
    "panda_joint6": 1.5707,
    "panda_joint7": 0.7853
}

urdf_file_path = os.path.join(urdf_path, args.file)
# mesh_path = os.path.join(urdf_path, 'meshes')

robot = urdfpy.URDF.load(urdf_file_path)

for joint in robot.actuated_joints:
    print(joint.name)


if args.a:
    robot.animate(use_collision=args.c)
else:
    robot.show(
        cfg=robot_default_dof_pos,
        use_collision=args.c)
