from pathlib import Path
import os
import isaacgym 
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from storm_kit.rl.agents import BCAgent, SACAgent, MPCAgent, MPOAgent, MPQAgent
from storm_kit.rl.policies import MPCPolicy, GaussianPolicy
from storm_kit.rl.value_functions import QFunction, TwinQFunction
from storm_kit.rl.world_models import GaussianWorldModel
from storm_kit.rl.replay_buffer import ReplayBuffer
from storm_kit.rl.util import Log


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    import isaacgymenvs
    import storm_kit.envs

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

    envs.render()

    base_dir = Path('./tmp_results/{}/{}'.format(cfg.task_name, cfg.train.agent.name))
    log_dir = os.path.join(base_dir, 'logs')
    model_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = Log(log_dir, cfg)
    log(f'Log dir: {log.dir}')
    writer = SummaryWriter(log.dir)


    # cfg.mpc.env_control_space = cfg.task.env.controlSpace
    # cfg.mpc.world = cfg.task.world
    # cfg.mpc.control_dt = cfg.task.sim.dt

    obs_dim = envs.obs_space.shape[0]
    act_dim = envs.action_space.shape[0]
    
    buffer = ReplayBuffer(capacity=int(2e6), obs_dim=obs_dim, act_dim=act_dim, device=torch.device('cpu'))
    # if cfg.train.expert_data_path is not None:
    #     data_path = os.path.abspath(cfg.train.expert_data_path)
    #     buffer.load(data_path)
    #     print('Buffer Loaded. Size = {}'.format(len(buffer)))
    # agent = BCAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
    #                  buffer=buffer, policy=policy, logger=log, tb_writer=writer, device=cfg.rl_device)


    if cfg.train.agent.name == 'SAC':
        policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, device=cfg.rl_device)
        critic = TwinQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
        agent = SACAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
                        buffer=buffer, policy=policy, critic=critic, logger=log, tb_writer=writer, device=cfg.rl_device)
    elif cfg.train.agent.name == 'MPO':
        policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, device=cfg.rl_device)
        critic = QFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
        agent = MPOAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
                        buffer=buffer, policy=policy, critic=critic, logger=log, tb_writer=writer, device=cfg.rl_device)
    elif cfg.train.agent.name == 'MPQ':
        policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, device=cfg.rl_device)
        critic = TwinQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
        world_model = GaussianWorldModel(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.world_model, device=cfg.rl_device)
        agent = MPQAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
                        buffer=buffer, policy=policy, critic=critic, world_model=world_model, logger=log, tb_writer=writer, device=cfg.rl_device)

    else:
        raise NotImplementedError('Invalid agent type: {}'.format(cfg.train.agent.name))

    agent.train(model_dir=model_dir)

if __name__ == "__main__":
    main()