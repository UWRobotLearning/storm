from pathlib import Path
import os
import isaacgym 
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from storm_kit.rl.agents import BCAgent, SACAgent, MPCAgent
from storm_kit.rl.policies import MPCPolicy, GaussianPolicy
from storm_kit.rl.replay_buffer import ReplayBuffer
from storm_kit.rl.util import Log


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    import isaacgymenvs
    from storm_kit.gym import tasks

    envs = isaacgymenvs.make(
        cfg.seed, 
        cfg.task_name, 
        cfg.task.env.numEnvs, 
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
        # **kwargs,
    )

    base_dir = Path('./tmp_results/{}/{}/eval'.format(cfg.task_name, cfg.train.agent.name))
    log_dir = os.path.join(base_dir, 'logs')
    checkpoint_dir = os.path.abspath(cfg.train.eval.checkpoint)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = Log(log_dir, cfg)
    log(f'Log dir: {log.dir}')
    writer = SummaryWriter(log.dir)


    cfg.mpc.env_control_space = cfg.task.env.controlSpace
    cfg.mpc.world = cfg.task.world
    cfg.mpc.control_dt = cfg.task.sim.dt

    obs_dim = envs.obs_space.shape[0]
    act_dim = envs.action_space.shape[0]
    
    buffer = ReplayBuffer(capacity=int(1e6), obs_dim=obs_dim, act_dim=act_dim, device=cfg.rl_device)
    policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, device=cfg.rl_device)
    agent = BCAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
                     buffer=buffer, policy=policy, logger=log, tb_writer=writer, device=cfg.rl_device)
    #load checkpoint
    agent.load(checkpoint_dir)
    eval_metrics = agent.evaluate(num_eval_episodes=cfg.train.eval.num_eval_episodes)
    print(eval_metrics)
    
if __name__ == "__main__":
    main()