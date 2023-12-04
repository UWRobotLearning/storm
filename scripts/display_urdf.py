import os, sys
import argparse

# from urdf_parser_py.urdf import URDF
from storm_kit.util_file import get_urdf_path
# from robomeshcat import Robot, Scene
import urdfpy 

urdf_path = get_urdf_path()

parser = argparse.ArgumentParser(usage='Load an URDF file')
parser.add_argument('file', type=str, nargs='?',
                    default='franka_description/franka_panda_curobo.urdf', help='File to load. Use - for stdin')
# parser.add_argument('-o', '--output', type=argparse.FileType('w'),
#                     default=None, help='Dump file to XML')
parser.add_argument('-a', action='store_true',
                    help='Visualize robot articulation')
parser.add_argument('-c', action='store_true',
                    help='Use collision geometry')

args = parser.parse_args()

# if args.file is None:
#     robot = URDF.from_parameter_server()
# else:
#     robot = URDF.from_xml_string(args.file.read())

# print(robot)

# if args.output is not None:
#     args.output.write(robot.to_xml_string())
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
mesh_path = os.path.join(urdf_path, 'meshes')
# scene = Scene()

# print(mesh_path)

"Create the first robot and add it to the scene"
# rob = Robot(urdf_path=urdf_file_path, name='visual')
# scene.add_robot(rob)

robot = urdfpy.URDF.load(urdf_file_path)

for joint in robot.actuated_joints:
    print(joint.name)


if args.a:
    robot.animate(use_collision=args.c)
else:
    robot.show(
        cfg = robot_default_dof_pos,
        use_collision=args.c)