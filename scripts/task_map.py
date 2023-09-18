from storm_kit.envs import PointRobotEnv, FrankaEnv
from storm_kit.mpc.rollout import PointRobotReacher, ArmReacher, PointRobotPusher, ArmPusher

task_map = {}
task_map['PointRobotReacher'] = {
    'env_cls': PointRobotEnv,
    'task_cls': PointRobotReacher,
}
task_map['FrankaReacher'] = {
    'env_cls': FrankaEnv,
    'task_cls': ArmReacher
}
task_map['PointRobotPusher'] = {
    'env_cls': PointRobotEnv,
    'task_cls': PointRobotPusher
}
task_map['FrankaPusher'] = {
    'env_cls': FrankaEnv,
    'task_cls': ArmPusher
}
