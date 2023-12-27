from pathlib import Path
import copy
import os
import isaacgym 
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
torch.manual_seed(0)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from storm_kit.learning.agents import BPAgent, SACAgent, MPCAgent, MPOAgent, MPQAgent
from storm_kit.learning.policies import MPCPolicy, GaussianPolicy, JointControlWrapper
from storm_kit.learning.value_functions import QFunction, TwinQFunction, EnsembleQFunction
from storm_kit.learning.world_models import GaussianWorldModel
from storm_kit.learning.replay_buffers import RobotBuffer
from storm_kit.learning.learning_utils import Log, buffer_from_file
from storm_kit.util_file import get_data_path
from storm_kit.learning.learning_utils import episode_runner
from task_map import task_map


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    
    from storm_kit.envs import IsaacGymRobotEnv
    task_details = task_map[cfg.task.name]
    task_cls = task_details['task_cls']    

    base_dir = Path('./tmp_results/{}/{}'.format(cfg.task_name, cfg.train.agent.name))
    log_dir = os.path.join(base_dir, 'logs')
    model_dir = os.path.join(base_dir, 'models')
    eval_rng = torch.Generator(device=cfg.rl_device)
    eval_rng.manual_seed(cfg.seed)
    train_rng = torch.Generator(device=cfg.rl_device)
    train_rng.manual_seed(cfg.seed)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = Log(log_dir, cfg)
    log(f'Log dir: {log.dir}')
    writer = SummaryWriter(log.dir)

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
    #Creating another copy for now. Might remove later
    policy_task = task_cls(
        cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)

    obs_dim = task.obs_dim
    act_dim = task.action_dim
    act_lows, act_highs = task.action_lims

    buffer = RobotBuffer(capacity=int(cfg.train.agent.max_buffer_size), device=cfg.rl_device)

    policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, task=policy_task, act_lows=act_lows, act_highs=act_highs, device=cfg.rl_device)
    #Load pretrained policy if required
    load_pretrained_policy = cfg.train.agent.load_pretrained_policy and (cfg.train.pretrained_policy_path is not None)
    pretrained_policy_loaded = False
    if load_pretrained_policy:
        checkpoint_path = Path('./tmp_results/{}/{}'.format(cfg.task_name, cfg.eval.pretrained_policy))
        print('Loading pre-trained policy from {}'.format(checkpoint_path))
        try:
            checkpoint = torch.load(checkpoint_path)
            policy_state_dict = checkpoint['policy_state_dict']
            remove_prefix = 'policy.'
            policy_state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in policy_state_dict.items()}
            policy.load_state_dict(policy_state_dict)
            pretrained_policy_loaded = True
            print('Pre-trained policy loaded')
        except:
            print('Policy loading failed')
    
    policy = JointControlWrapper(config=cfg.task.joint_control, policy=policy, device=cfg.rl_device)
    random_ensemble_q = cfg.train.agent.get('random_ensemble_q', False)
    
    #Note: Later we will make this a single class
    if cfg.train.critic.ensemble_size == 1:
        critic = QFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
    elif cfg.train.critic.ensemble_size == 2:
        critic = TwinQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
    else:
        critic = EnsembleQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device) 
    target_critic = copy.deepcopy(critic)

    num_pretrain_steps = cfg.train.agent.get('num_pretrain_steps', 0)
    if num_pretrain_steps > 0 and (not pretrained_policy_loaded):
        try:
            base_dir = os.path.abspath('./tmp_results/{}'.format(cfg.task_name))
            data_path = os.path.join(base_dir, cfg.train.dataset_path)
            init_buffer, _ = buffer_from_file(data_path)
        except Exception as e:
            print('Could not load data', e)
        print('Pretraining')
        pretrain_agent = BPAgent(
            cfg.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim, 
            buffer=init_buffer, policy=policy, critic=critic, runner_fn=episode_runner, 
            logger=log, tb_writer=writer, device=cfg.rl_device, eval_rng=eval_rng)
        pretrain_agent.train(model_dir=model_dir)
        print('Pretraining done')
    
    agent_name = cfg.train.agent.name
    
    if  agent_name == 'SAC':
        agent = SACAgent(cfg.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim, 
                         buffer=buffer, policy=policy, critic=critic, runner_fn=episode_runner, target_critic=target_critic, 
                         logger=log, tb_writer=writer, device=cfg.rl_device, train_rng=train_rng, eval_rng=eval_rng)
    
    elif agent_name == 'MPO':
        agent = MPOAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
                        buffer=buffer, policy=policy, critic=critic, logger=log, tb_writer=writer, device=cfg.rl_device)
    
    elif agent_name == 'MPQ': 
        dyn_model_cls = task_details['dynamics_model_cls']
        mpc_policy = MPCPolicy(
            obs_dim=obs_dim, act_dim=act_dim, config=cfg.mpc, 
            sampling_policy=policy.policy, value_function=critic, 
            task_cls=task_cls, dynamics_model_cls=dyn_model_cls,
            device=cfg.rl_device) 
        mpc_policy = JointControlWrapper(config=cfg.task.joint_control, policy=mpc_policy, device=cfg.rl_device)
        #For target mpc policy, we need to set num_instances to the train batch size
        target_mpc_cfg = cfg.mpc
        target_mpc_cfg['mppi']['state_batch_size'] = cfg.train.agent.train_batch_size
        target_mpc_policy = MPCPolicy(
            obs_dim=obs_dim, act_dim=act_dim, config=target_mpc_cfg, 
            sampling_policy=policy.policy, value_function=critic, 
            task_cls=task_cls, dynamics_model_cls=dyn_model_cls,
            device=cfg.rl_device) 
        # world_model = GaussianWorldModel(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.world_model, device=cfg.rl_device)
        agent = MPQAgent(
            cfg.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim,
            buffer=buffer, policy=policy, mpc_policy=mpc_policy, target_mpc_policy=target_mpc_policy,
            critic=critic, runner_fn=episode_runner, 
            target_critic=target_critic, logger=log, tb_writer=writer, device=cfg.rl_device)

    if agent_name in ['SAC', 'MPO', 'MPQ']:
        print('Training {} agent'.format(cfg.train.agent.name))
        agent.train(model_dir=model_dir)
    # else:
        # raise NotImplementedError('Invalid agent type: {}'.format(cfg.train.agent.name))
    

if __name__ == "__main__":
    main()