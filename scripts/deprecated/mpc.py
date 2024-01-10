
from pathlib import Path
import isaacgym 
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import hydra
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
from storm_kit.learning.policies import MPCPolicy, JointControlWrapper
from storm_kit.learning.agents import MPCAgent
from storm_kit.learning.replay_buffers import RobotBuffer
from storm_kit.learning.learning_utils import episode_runner
from task_map import task_map



@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)
    from storm_kit.envs import IsaacGymRobotEnv
    task_details = task_map[cfg.task.name]
    task_cls = task_details['task_cls']   
    
    base_dir = Path('./tmp_results/{}/{}'.format(cfg.task_name, 'MPC'))
    model_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data')
    eval_rng = torch.Generator(device=cfg.rl_device)

    if model_dir is not None:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    if data_dir is not None:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    #Initialize environment
    envs = IsaacGymRobotEnv(
        cfg.task, 
        cfg.rl_device, 
        cfg.sim_device, 
        cfg.graphics_device_id, 
        cfg.headless, 
        False, 
        cfg.force_render
    )
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
    # agent = MPCAgent(cfg.eval, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim, 
    #                  buffer=buffer, policy=policy, runner_fn=episode_runner, device=cfg.rl_device)
    st=time.time()
    # metrics = agent.collect_experience(num_episodes=num_episodes, update_buffer=True, deterministic=deterministic_eval, debug=False)
    num_episodes = cfg.eval.num_episodes
    deterministic_eval = cfg.eval.deterministic_eval
    print('Collecting {} episodes. Deterministic = {}'.format(num_episodes, deterministic_eval))
    metrics = episode_runner(
        envs,
        num_episodes = num_episodes, 
        policy = policy,
        task = task,
        buffer = buffer,
        deterministic = deterministic_eval,
        debug = False,
        device = cfg.rl_device,
        rng = eval_rng)
    print(metrics)

    # if logger is not None:
    #     logger.row(metrics)
    # if tb_writer is not None:
    #     for k, v in metrics.items():
    #         self.tb_writer.add_scalar('Eval/' + k, v, 0)
    # print(metrics)
    
    print('Time taken = {}'.format(time.time() - st))
    data_dir = data_dir if cfg.eval.save_buffer else None
    # if model_dir is not None:
    #     print('Saving agent to {}'.format(model_dir))
    if data_dir is not None:
        print('Saving buffer to {}'.format(data_dir))
        buffer.save(os.path.join(data_dir, 'agent_buffer_0.pt'))

    # agent.save(model_dir, data_dir)
    print('Data saved')

if __name__ == "__main__":
    main()