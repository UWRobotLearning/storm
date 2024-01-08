import rospy
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
import os
import hydra
import torch
torch.manual_seed(0)
from torch.profiler import profile, record_function, ProfilerActivity
import time
from storm_kit.learning.policies import MPCPolicy, JointControlWrapper
from storm_kit.tasks import ArmReacher
from storm_kit.learning.agents import MPCAgent
from storm_kit.learning.replay_buffers import RobotBuffer
from storm_kit.learning.learning_utils import episode_runner, minimal_episode_runner

task_map = {}
task_map['FrankaReacherRealRobot'] = {
    'task_cls': ArmReacher
}
task_map['FrankaTrayReacherRealRobot'] = {
    'task_cls': ArmReacher
}

@hydra.main(config_name="config", config_path="../../content/configs/gym")
def main(cfg: DictConfig):

    torch.set_default_dtype(torch.float32)
    from franka_real_robot_env import FrankaRealRobotEnv

    rospy.init_node("mpc_robot", anonymous=True, disable_signals=True)    

    task_details = task_map[cfg.task.name]
    task_cls = task_details['task_cls']   
    
    base_dir = Path('../tmp_results/{}/{}'.format(cfg.task_name, 'MPC'))
    model_dir = os.path.abspath(os.path.join(base_dir, 'models'))
    data_dir = os.path.abspath(os.path.join(base_dir, 'data'))

    if model_dir is not None:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    if data_dir is not None:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    #Initialize environment
    envs = FrankaRealRobotEnv(
        cfg.task, 
        cfg.rl_device)
    #Initialize task
    task = task_cls(
        cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)

    #Initialize MPC Policy
    obs_dim = task.obs_dim
    act_dim = task.action_dim
    buffer = RobotBuffer(capacity=int(1e6), device=cfg.rl_device)   
    policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.mpc, task_cls=task_cls, device=cfg.rl_device)
    policy = JointControlWrapper(config=cfg.task.joint_control, policy=policy, device=cfg.rl_device)
    #Initialize Agent
    agent = MPCAgent(cfg.eval, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim, 
                     buffer=buffer, policy=policy, runner_fn=minimal_episode_runner, device=cfg.rl_device)
    st=time.time()
    num_episodes = cfg.eval.num_episodes
    deterministic_eval = cfg.eval.deterministic_eval
    print('Collecting {} episodes. Deterministic = {}'.format(num_episodes, deterministic_eval))
    metrics = agent.collect_experience(num_episodes=num_episodes, update_buffer=True, deterministic=deterministic_eval, debug=False)
    print(metrics)
    
    print('Time taken = {}'.format(time.time() - st))
    data_dir = data_dir if cfg.eval.save_buffer else None
    if model_dir is not None:
        print('Saving agent to {}'.format(model_dir))
    if data_dir is not None:
        print('Saving buffer to {}'.format(data_dir))
    agent.save(model_dir, data_dir)
    print('Agent saved')

if __name__ == "__main__":
    main()