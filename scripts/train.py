from pathlib import Path
import os
import isaacgym 
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from storm_kit.learning.agents import BPAgent, SACAgent, MPCAgent, MPOAgent, MPQAgent
from storm_kit.learning.policies import MPCPolicy, GaussianPolicy, JointControlWrapper
from storm_kit.learning.value_functions import QFunction, TwinQFunction, EnsembleQFunction
from storm_kit.learning.world_models import GaussianWorldModel
from storm_kit.learning.replay_buffers import ReplayBuffer, RobotBuffer
from storm_kit.learning.learning_utils import Log, buffer_from_folder
from storm_kit.util_file import get_data_path
from storm_kit.learning.learning_utils import episode_runner
from task_map import task_map


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    
    task_details = task_map[cfg.task.name]
    env_cls = task_details['env_cls']
    task_cls = task_details['task_cls']
    envs = env_cls(
        cfg.task, 
        cfg.rl_device, 
        cfg.sim_device, 
        cfg.graphics_device_id, 
        cfg.headless, 
        False, 
        cfg.force_render
    )

    #Initialize task
    task = task_cls(cfg=cfg.task.rollout, device=cfg.rl_device, world_params=cfg.task.world)
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

    obs_dim = task.obs_dim
    act_dim = task.action_dim
    act_lows, act_highs = task.action_lims

    buffer = RobotBuffer(capacity=int(cfg.train.agent.max_buffer_size), device=cfg.rl_device)

    if cfg.train.agent.name == "BP":
        try:
            data_path = os.path.join(get_data_path(), cfg.train.expert_data_path)
            buffer = buffer_from_folder(data_path)
            print(buffer)
        except Exception as e:
            print(e)
        
        agent = BPAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
                        buffer=buffer, policy=policy, logger=log, tb_writer=writer, device=cfg.rl_device)
        exit()

    
    if cfg.train.agent.name == 'SAC':
        # buffer = ReplayBuffer(capacity=int(2e6), obs_dim=obs_dim, act_dim=act_dim, device=torch.device('cpu'))
        policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, device=cfg.rl_device)
        policy = JointControlWrapper(config=cfg.task, policy=policy, act_lows=act_lows, act_highs=act_highs, device=cfg.rl_device)
        if cfg.train.agent.random_ensemble_q:
            critic = EnsembleQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device) 
        else:
            critic = TwinQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
        agent = SACAgent(cfg.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim, 
                         buffer=buffer, policy=policy, critic=critic, runner_fn=episode_runner, 
                         logger=log, tb_writer=writer, device=cfg.rl_device)
    
    elif cfg.train.agent.name == 'MPO':
        buffer = ReplayBuffer(capacity=int(2e6), obs_dim=obs_dim, act_dim=act_dim, device=torch.device('cpu'))
        policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, device=cfg.rl_device)
        critic = QFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
        agent = MPOAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
                        buffer=buffer, policy=policy, critic=critic, logger=log, tb_writer=writer, device=cfg.rl_device)
    
    elif cfg.train.agent.name == 'MPQ':
        nn_policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.nn_policy, device=cfg.rl_device)
        critic = TwinQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
        mpc_policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.task.mpc, rollout_cls=task_cls, value_function=critic, device=cfg.rl_device)
        target_mpc_policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.target_mpc_policy, rollout_cls=task_cls, value_function=critic, device=cfg.rl_device)
        # world_model = GaussianWorldModel(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.world_model, device=cfg.rl_device)
        agent = MPQAgent(cfg.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim,
                        buffer=buffer, policy=nn_policy, mpc_policy=mpc_policy, target_mpc_policy=target_mpc_policy, critic=critic, runner_fn=episode_runner, 
                        logger=log, tb_writer=writer, device=cfg.rl_device)

    else:
        raise NotImplementedError('Invalid agent type: {}'.format(cfg.train.agent.name))

    agent.train(model_dir=model_dir)

if __name__ == "__main__":
    main()