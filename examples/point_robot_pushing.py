
from pathlib import Path
import isaacgym 
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import hydra
import torch
torch.manual_seed(0)
from torch.profiler import profile, record_function, ProfilerActivity
import time
from storm_kit.learning.policies import MPCPolicy
from storm_kit.learning.agents import MPCAgent
from storm_kit.learning.replay_buffers import ReplayBuffer, RobotBuffer
from storm_kit.learning.learning_utils import episode_runner


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    # import isaacgymenvs
    # import storm_kit.envs
    from storm_kit.envs import PointRobotEnv
    from storm_kit.mpc.rollout import PointRobotPusher
    # envs = isaacgymenvs.make(
    #     cfg.seed, 
    #     cfg.task_name, 
    #     cfg.task.env.numEnvs, 
    #     cfg.sim_device,
    #     cfg.rl_device,
    #     cfg.graphics_device_id,
    #     cfg.headless,
    #     cfg.multi_gpu,
    #     cfg.capture_video,
    #     cfg.force_render,
    #     cfg,
    #     # **kwargs,
    # )
    #Initialize franka env
    envs = PointRobotEnv(
        cfg.task, 
        cfg.rl_device, 
        cfg.sim_device, 
        cfg.graphics_device_id, 
        cfg.headless, 
        False, 
        cfg.force_render
    )

    #Initialize task
    task = PointRobotPusher(cfg=cfg.task.rollout, world_params=cfg.task.world, device=cfg.rl_device)
    base_dir = Path('./tmp_results/{}/{}'.format(cfg.task_name, cfg.task.train.agent.name))
    model_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    cfg.task.mpc.world = cfg.task.world
    # cfg.task.mpc.control_dt = cfg.task.sim.dt

    obs_dim = task.obs_dim
    act_dim = task.action_dim

    buffer = RobotBuffer(capacity=int(1e6), n_dofs=cfg.task.rollout.n_dofs, 
                         obs_dim=obs_dim, device=cfg.rl_device)   
    policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.task.mpc, rollout_cls=PointRobotPusher, device=cfg.rl_device)
    agent = MPCAgent(cfg.task.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim, 
                     buffer=buffer, policy=policy, runner_fn=episode_runner, device=cfg.rl_device)
    st=time.time()
    metrics = agent.collect_experience(num_episodes=cfg.task.train.agent.num_episodes, update_buffer=True)
    print(metrics)
    print('Time taken = {}'.format(time.time() - st))
    print('Saving agent. Buffer save = {}'.format(cfg.train.agent.save_buffer))
    agent.save(model_dir, save_buffer=cfg.train.agent.save_buffer)
    print('Agent saved')





if __name__ == "__main__":
    main()