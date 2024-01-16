from pathlib import Path
import copy
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
from storm_kit.learning.replay_buffer import ReplayBuffer
from storm_kit.learning.learning_utils import Log, buffer_from_file
from storm_kit.util_file import get_data_path
from storm_kit.learning.learning_utils import episode_runner
from task_map import task_map

def init_logging(task_name, agent_name, cfg):
    base_dir = Path('./tmp_results/{}/{}'.format(task_name, agent_name))
    log_dir = os.path.join(base_dir, 'logs')
    model_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = Log(log_dir, cfg)
    log(f'Log dir: {log.dir}')
    writer = SummaryWriter(log.dir)  

    return {
        'log_dir': log_dir,
        'model_dir': model_dir,
        'log': log,
        'tb_writer': writer,
    }


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    
    task_details = task_map[cfg.task.name]
    task_cls = task_details['task_cls']    

    logging_info = init_logging(cfg.task_name, cfg.train.agent.name, cfg)
    model_dir = logging_info['model_dir']
    log = logging_info['log']
    writer = logging_info['tb_writer']
    eval_rng = torch.Generator(device=cfg.rl_device)
    eval_rng.manual_seed(cfg.seed)
    train_rng = torch.Generator(device=cfg.rl_device)
    train_rng.manual_seed(cfg.seed)

    #Initialize environment
    if not cfg.real_robot_exp:
        from storm_kit.envs import IsaacGymRobotEnv
        envs = IsaacGymRobotEnv(
            cfg.task, 
            cfg.rl_device, 
            cfg.sim_device, 
            cfg.graphics_device_id, 
            cfg.headless, 
            False, 
            cfg.force_render
        )
    else:
        from storm_kit.envs.panda_real_robot_env import PandaRealRobotEnv
        envs = PandaRealRobotEnv(
            cfg.task,
            device=cfg.rl_device,
            headless=cfg.headless,
            safe_mode=False
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
    state_bounds = task.state_bounds

    buffer = ReplayBuffer(capacity=int(cfg.train.agent.max_buffer_size), device=cfg.rl_device)

    train_from_scratch = cfg.train.agent.get('train_from_scratch', True)
    #Load init data
    load_init_data = cfg.train.agent.get('load_init_data', False) or train_from_scratch or (agent_name == "BP")
    init_data_loaded = False
    init_buffer = None
    if load_init_data:
        try:
            base_dir = os.path.abspath('./tmp_results/{}'.format(cfg.task_name))
            data_path = os.path.join(base_dir, cfg.train.dataset_path)
            init_buffer, _ = buffer_from_file(data_path)
            init_data_loaded = True
            print('Loaded init dataset of length = {}'.format(len(init_buffer)))
        except Exception as e:
            print('Could not load data', e)
 
    policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, task=policy_task, act_lows=act_lows, act_highs=act_highs, device=cfg.rl_device)
    #Load pretrained 
    # load_pretrained = cfg.train.agent.load_pretrained and (cfg.train.pretrained_path is not None)
    load_pretrained = not train_from_scratch #cfg.train.agent.load_pretrained and (not train_from_scratch)
    # pretrained_loaded = False
    if load_pretrained:
        try:
            checkpoint_path = Path('./tmp_results/{}/{}'.format(cfg.task_name, cfg.train.pretrained_path))
            print('Loading pre-trained policy from {}'.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            policy_state_dict = checkpoint['policy_state_dict']
            remove_prefix = 'policy.'
            policy_state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in policy_state_dict.items()}
            policy.load_state_dict(policy_state_dict)
            # pretrained_loaded = True
            print('Pre-trained loaded')
        except:
            print('Pre-trained loading failed. Training from scratch...')
            train_from_scratch = True

    policy = JointControlWrapper(
        config=cfg.task.joint_control, 
        policy=policy, 
        state_bounds=state_bounds,
        device=cfg.rl_device)
    
    #Note: Later we will make this a single class
    if cfg.train.critic.ensemble_size == 1:
        critic = QFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
    elif cfg.train.critic.ensemble_size == 2:
        critic = TwinQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device)
    else:
        critic = EnsembleQFunction(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.critic, device=cfg.rl_device) 
    
    target_critic = copy.deepcopy(critic)
    #freeze target critic parameters
    for p in target_critic.parameters(): p.requires_grad = False
    #create mpc policy
    dyn_model_cls = task_details['dynamics_model_cls']
    # policy.policy
    mpc_policy = MPCPolicy(
        obs_dim=obs_dim, act_dim=act_dim, config=cfg.mpc, 
        sampling_policy=None, value_function=critic, 
        task_cls=task_cls, dynamics_model_cls=dyn_model_cls,
        device=cfg.rl_device) 
    
    mpc_policy = JointControlWrapper(
        config=cfg.task.joint_control, 
        policy=mpc_policy, 
        state_bounds=state_bounds,
        device=cfg.rl_device)

    agent_name = cfg.train.agent.name
    run_pretraining = (train_from_scratch and init_data_loaded) or (agent_name == "BP")
    run_pretraining = run_pretraining and (cfg.train.agent.num_pretrain_steps > 0)
    pretrain_critic = cfg.train.agent.pretrain_critic
    if run_pretraining:
        print('Running Behavior Pretraining')
        pretrain_log_info = init_logging(cfg.task_name, 'BP', cfg)
        pretrain_agent = BPAgent(
            cfg.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim, 
            buffer=init_buffer, runner_fn=episode_runner, 
            policy=policy, mpc_policy=mpc_policy,
            qf=critic if pretrain_critic else None, 
            # target_critic=target_critic if pretrain_critic else None,
            logger=pretrain_log_info['log'], tb_writer=writer, device=cfg.rl_device, eval_rng=eval_rng)
        pretrain_agent.train(model_dir=pretrain_log_info['model_dir'])
        print('Behavior Pretraining done')

    if  agent_name == 'SAC':
        agent = SACAgent(cfg.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim, 
                         buffer=buffer, policy=policy, critic=critic, runner_fn=episode_runner, target_critic=target_critic, 
                         logger=log, tb_writer=writer, device=cfg.rl_device, train_rng=train_rng, eval_rng=eval_rng)
    
    elif agent_name == 'MPO':
        agent = MPOAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
                        buffer=buffer, policy=policy, critic=critic, logger=log, tb_writer=writer, device=cfg.rl_device)
    
    elif agent_name == 'MPQ': 
        #For target mpc policy, we need to set num_instances to the train batch size
        target_mpc_cfg = cfg.mpc
        target_mpc_cfg['mppi']['state_batch_size'] = cfg.train.agent.train_batch_size
        # target_mpc_cfg['mppi']['horizon'] = 20
        # target_mpc_cfg['mppi']['n_iters'] = 2
        # target_mpc_cfg['mppi']['cl_act_frac'] = 0.0
        target_mpc_policy = MPCPolicy(
            obs_dim=obs_dim, act_dim=act_dim, config=target_mpc_cfg, 
            sampling_policy=None, value_function=critic, 
            task_cls=task_cls, dynamics_model_cls=dyn_model_cls,
            device=cfg.rl_device) 
        
        agent = MPQAgent(
            cfg.train.agent, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim,
            buffer=buffer, policy=policy, mpc_policy=mpc_policy, target_mpc_policy=target_mpc_policy,
            critic=critic, runner_fn=episode_runner, target_critic=target_critic,
            init_buffer=init_buffer, logger=log, tb_writer=writer, device=cfg.rl_device,
            train_rng=train_rng, eval_rng=eval_rng)

    if agent_name in ['SAC', 'MPO', 'MPQ']:
        print('Training {} agent'.format(cfg.train.agent.name))
        print(cfg.debug)
        agent.train(model_dir=model_dir, debug=cfg.debug)
    

if __name__ == "__main__":
    main()