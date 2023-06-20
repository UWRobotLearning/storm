from .franka_env import FrankaEnv
from .franka_reacher import FrankaReacher
from .franka_pusher import FrankaPusher

from isaacgymenvs.tasks import isaacgym_task_map

isaacgym_task_map['FrankaReacher'] = FrankaReacher
isaacgym_task_map['FrankaPusher'] = FrankaPusher