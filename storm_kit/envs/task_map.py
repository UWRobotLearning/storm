# from storm_kit.mpc.rollout import PointRobotReacher, ArmReacher, PointRobotPusher, ArmPusher
from storm_kit.tasks import ArmReacher
from storm_kit.mpc.model import URDFKinematicModel
task_map = {}
# task_map['PointRobotReacher'] = {
#     'task_cls': PointRobotReacher,
# }
task_map['FrankaReacher'] = {
    'task_cls': ArmReacher,
    'dynamics_model_cls': URDFKinematicModel
}
task_map['FrankaReacherRealRobot'] = {
    'task_cls': ArmReacher,
    'dynamics_model_cls': URDFKinematicModel
}

# task_map['PointRobotPusher'] = {
#     'task_cls': PointRobotPusher
# }
# task_map['FrankaPusher'] = {
#     'task_cls': ArmPusher
# }
task_map['FrankaTrayReacher'] = {
    'task_cls': ArmReacher,
    'dynamics_model_cls': URDFKinematicModel
}
# task_map['TrayReacher'] = {
#     'task_cls': TrayReacher
# }