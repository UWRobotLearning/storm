
import os
import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
from storm_kit.mpc.task import ReacherTask
from storm_kit.util_file import get_urdf_path, get_gym_configs_path
import torch.autograd.profiler as profiler

robot_file =  'franka.yml'
task_file =  'franka_reacher.yml'
world_file = 'collision_primitives_3d.yml'
world_yml = os.path.join(get_gym_configs_path(), world_file)
robot_yml = os.path.join(get_gym_configs_path(),'franka.yml')

if(torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'

device = torch.device('cuda', 0) 
tensor_args = {'device':device, 'dtype':torch.float32}

# get camera data:
mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
mpc_control.update_params(goal_state = np.zeros((22,)))
t_step = 0.0
current_robot_state = dict(position=np.zeros(7),
                    velocity=np.zeros(7),
                    acceleration=np.zeros(7))
with profiler.profile(profile_memory=True, use_cuda=True) as prof:
    command = mpc_control.get_command(t_step, current_robot_state, control_dt=0.001, WAIT=True)
print(prof.key_averages(group_by_stack_n=5).table(sort_by='cuda_time_total'))