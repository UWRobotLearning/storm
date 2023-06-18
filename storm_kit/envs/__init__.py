from .franka_env import FrankaEnv
from .franka_reacher import FrankaReacher

from isaacgymenvs.tasks import isaacgym_task_map

isaacgym_task_map['FrankaReacher'] = FrankaReacher