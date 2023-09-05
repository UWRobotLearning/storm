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
from storm_kit.learning.value_functions import QFunction, TwinQFunction
from storm_kit.learning.world_models import GaussianWorldModel
from storm_kit.learning.replay_buffers import ReplayBuffer, RobotBuffer
from storm_kit.learning.learning_utils import Log, buffer_from_folder
from storm_kit.util_file import get_data_path
from storm_kit.learning.learning_utils import episode_runner

def get_env_and_task(cfg):
    task_name = cfg.task.name
    if task_name == 'FrankaReacher':
        from storm_kit.envs import FrankaEnv
        from storm_kit.mpc.rollout import ArmReacher
        env_cls = FrankaEnv
        task_cls = ArmReacher

    elif task_name == 'PointRobotPusher':
        from storm_kit.envs import PointRobotEnv
        from storm_kit.mpc.rollout import PointRobotPusher
        env_cls = PointRobotEnv
        task_cls = PointRobotPusher
    
    return env_cls, task_cls


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)

    env_cls, task_cls = get_env_and_task(cfg)

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

    # cfg.mpc.world = cfg.task.world
    # cfg.mpc.control_dt = cfg.task.sim.dt

    obs_dim = task.obs_dim
    act_dim = task.action_dim

    buffer = RobotBuffer(capacity=int(1e6), device=cfg.rl_device)
    # cfg.mpc.env_control_space = cfg.task.env.controlSpace
    # cfg.mpc.world = cfg.task.world
    # cfg.mpc.control_dt = cfg.task.sim.dt

    # obs_dim = envs.obs_space.shape[0]
    # act_dim = envs.action_space.shape[0]

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
        policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, rollout_cls=None, device=cfg.rl_device)
        # policy = JointControlWrapper(config=cfg.task, policy=policy, device=cfg.rl_device)
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
        # buffer = ReplayBuffer(capacity=int(2e6), obs_dim=obs_dim, act_dim=act_dim, device=torch.device('cpu'))
        # policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, device=cfg.rl_device)
        critic = TwinQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
        policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.task.mpc, rollout_cls=task_cls, value_function=critic, device=cfg.rl_device)
        print('initializing target policy')
        target_policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.target_mpc_policy, rollout_cls=task_cls, value_function=critic, device=cfg.rl_device)
        # world_model = GaussianWorldModel(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.world_model, device=cfg.rl_device)
        agent = MPQAgent(cfg.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim,
                        buffer=buffer, policy=policy, critic=critic, target_policy=target_policy, runner_fn=episode_runner, 
                        logger=log, tb_writer=writer, device=cfg.rl_device)

    else:
        raise NotImplementedError('Invalid agent type: {}'.format(cfg.train.agent.name))

    agent.train(model_dir=model_dir)

if __name__ == "__main__":
    main()