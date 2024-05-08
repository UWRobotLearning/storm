import copy
from collections import defaultdict
from pathlib import Path
from typing import Union, Tuple
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf, DictConfig
import time 
import gym
# import d4rl
import numpy as np
import isaacgym
import isaacgymenvs
import torch
from torch.profiler import record_function
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from storm_kit.envs.gym_env_wrapper import GymEnvWrapper
from storm_kit.learning.policies import MPCPolicy, GaussianPolicy
from storm_kit.learning.value_functions import TwinQFunction, ValueFunction, EnsembleQFunction, EnsembleValueFunction
from storm_kit.learning.learning_utils import Log, evaluate_policy, preprocess_dataset, dict_to_device, buffer_dict_from_folder #, return_range
from storm_kit.learning.agents import BPAgent
from storm_kit.learning.replay_buffer import ReplayBuffer, qlearning_dataset, qlearning_dataset2
from task_map import task_map
import wandb


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
    if cfg.wandb_activate:
        # wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.tensorboard.patch(root_logdir=log_dir, pytorch=True, tensorboard_x=False)
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, sync_tensorboard=True)

    writer = SummaryWriter(log.dir)  

    return {
        'log_dir': log_dir,
        'model_dir': model_dir,
        'log': log,
        'tb_writer': writer,
    }

def normalize_score(task_name, returns):
    if task_name.startswith(('hopper', 'walker2d', 'halfcheetah', 'antmaze')):
        normalized_score= d4rl.get_normalized_score(task_name, returns)*100.0
    else:
        normalized_score = returns
    return normalized_score

def get_d4rl_dataset(env):
    from urllib.error import HTTPError
    while True:
        try:
            return env.get_dataset()
        except (HTTPError, OSError):
            print('Unable to download dataset. Retry.')

def get_task_and_dataset(task_name:str, cfg=None): #log max_episode_steps
    if task_name.startswith(('hopper', 'walker2d', 'halfcheetah', 'antmaze')):
        env = gym.make(task_name)
        task = None
        dataset = get_d4rl_dataset(env)
        # min_ret, max_ret = return_range(dataset, 1000)

        num_points = len(dataset['actions'])
        replay_buffer =  ReplayBuffer(capacity=num_points)
        replay_buffer.add_batch(dataset)
        #checking data is properly transformed
        for k, v in replay_buffer.items():
            assert torch.allclose(torch.as_tensor(dataset[k], dtype=torch.float32), v)
        # dataset = qlearning_dataset2(env)
        # for k in dataset.keys():
        #     print(k, dataset[k].shape, replay_buffer[k].shape)
        #     assert(torch.allclose(torch.as_tensor(dataset[k]), torch.as_tensor(replay_buffer[k])))
        
        # if any(s in task_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        #     min_ret, max_ret = return_range(dataset, max_episode_steps)
        #     log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        #     dataset['rewards'] /= (max_ret - min_ret)
        #     dataset['rewards'] *= max_episode_steps
        # elif 'antmaze' in task_name:
        #     dataset['rewards'] -= 1.

        # for k, v in dataset.items():
        #     dataset[k] = torchify(v)
    else:
        task_details = task_map[task_name]
        task_cls = task_details['task_cls']    
        #Initialize environment
        if not cfg.real_robot_exp:
            from storm_kit.envs.isaac_gym_robot_env import IsaacGymRobotEnv
            env = IsaacGymRobotEnv(
                cfg.task.env, cfg.task.world, cfg.rl_device, 
                cfg.sim_device, cfg.graphics_device_id, 
                cfg.headless, False, cfg.force_render
            )
        else:
            from storm_kit.envs.panda_real_robot_env import PandaRealRobotEnv
            # env = PandaRealRobotEnv(
            #     cfg.task.env, cfg.task.world, device=cfg.rl_device,
            #     headless=cfg.headless, safe_mode=False
            # )
            env = PandaRealRobotEnv(
                cfg.task, device=cfg.rl_device,
                headless=cfg.headless, safe_mode=False
            )
        #Load dataset
        base_dir = Path('./tmp_results/{}/{}'.format(cfg.task_name, 'policy_eval'))
        data_dir = os.path.join(base_dir, 'data')

        #Initialize task
        task = task_cls(
            cfg=cfg.task.task, world_cfg=cfg.task.world, device=cfg.rl_device, viz_rollouts=False)

        #Load bufffers from folder
        buffer_dict = buffer_dict_from_folder(data_dir)
        replay_buffer = buffer_dict['mpc_buffer_50ep_no_goal_obs.pt']
        
    return env, task, replay_buffer

def get_mpc_policy(cfg, policy=None, vf=None, qf=None, prediction_metrics=None):
    task_name = cfg.task_name
    task_details = task_map[task_name]
    task_cls = task_details['task_cls']    
    dynamics_model_cls = task_details['dynamics_model_cls']
    mpc_policy = MPCPolicy(
        obs_dim=1, act_dim=1, config=cfg.mpc,
        task_cls=task_cls, dynamics_model_cls=dynamics_model_cls, 
        sampling_policy=policy, vf=vf, qf=qf,
        device=cfg.rl_device)
    # mpc_policy.set_prediction_metrics(prediction_metrics)
    return mpc_policy

@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env, task, dataset = get_task_and_dataset(cfg.task_name, cfg)
    print('Dataset obs shape:', dataset['observations'].shape)
    # exit()
    # import pdb; pdb.set_trace()
    train_dataset, val_dataset, terminal_dataset, dataset_info = preprocess_dataset(
        dataset, env, task, cfg.train.agent, 
        normalize_score_fn=lambda returns: normalize_score(cfg.task_name, returns))
    train_dataset = qlearning_dataset(train_dataset)
    if val_dataset is not None: val_dataset.to(cfg.rl_device)
    # if val_dataset is not None: val_dataset = qlearning_dataset(val_dataset)
    if terminal_dataset is not None: terminal_dataset.to(cfg.rl_device) #qlearning_dataset(terminal_dataset) #.to(cfg.rl_device)
    print(dataset_info)

    env = GymEnvWrapper(env, task)

    logging_info = init_logging(cfg.task_name, cfg.train.agent.name, cfg)
    model_dir = logging_info['model_dir']
    tb_writer = logging_info['tb_writer']
    eval_rng = torch.Generator(device=cfg.rl_device)
    eval_rng.manual_seed(cfg.seed)
    train_rng = torch.Generator(device=cfg.rl_device)
    train_rng.manual_seed(cfg.seed)
    try:
        env.seed(cfg.seed)
    except:
        print('Env does not have seed')
    obs_dim, act_dim = dataset['observations'].shape[-1], dataset['actions'].shape[-1]

    normalization_stats = {
        "disc_return_mean": 0.0, "disc_return_std": 1.0, 
        "Vmax": float('inf'), "Vmin": -float('inf'),
        "obs_mean": None, "obs_std": None, 
        "obs_max": float('inf'), "obs_min": -float('inf')}

    if cfg.train.agent.normalize_returns:
        normalization_stats["disc_return_mean"] = dataset_info["disc_return_mean"]
        normalization_stats["disc_return_std"] = dataset_info["disc_return_std"]
        normalization_stats['V_min'], normalization_stats['V_max'] = dataset_info['V_min'], dataset_info['V_max']
    
    if cfg.train.agent.normalize_observations:
        normalization_stats["obs_mean"] = dataset_info['obs_mean'].to(cfg.rl_device)
        normalization_stats["obs_std"] = dataset_info["obs_std"].to(cfg.rl_device)
        normalization_stats["obs_max"] = dataset_info['obs_max'].to(cfg.rl_device)
        normalization_stats["obs_min"] = dataset_info["obs_min"].to(cfg.rl_device)
        
    policy = GaussianPolicy(
        obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, device=cfg.rl_device) #task=task,
    # qf = TwinQFunction(
    #     obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.qf, device=cfg.rl_device)
    qf = EnsembleQFunction(
        obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.qf, device=cfg.rl_device)
    vf = EnsembleValueFunction(
        obs_dim=obs_dim, config=cfg.train.vf, device=cfg.rl_device)
    vf.set_normalization_stats(normalization_stats)

    #initialize mpc policy
    mpc_policy = get_mpc_policy(cfg, policy=None, vf=vf, qf=qf, prediction_metrics=normalization_stats)

    # Run behavior pretraining
    # num_train_steps = int(cfg.train.agent['num_pretrain_steps'])
    num_train_epochs = int(cfg.train.agent['num_train_epochs'])
    agent = BPAgent(
        cfg.train.agent, policy=policy, qf=None, vf=vf, 
        device=cfg.rl_device
    )

    log_metrics = defaultdict(list)
    pbar = tqdm(range(int(num_train_epochs)), desc='train')
    num_eval_episodes = cfg.train.agent.num_eval_episodes
    eval_first_iter = cfg.train.agent.eval_first_iter
    eval_freq = cfg.train.agent.eval_freq
    validation_freq = cfg.train.agent.validation_freq
    max_episode_steps = cfg.task.env.get('episodeLength', 1000)
    terminal_batch_size = cfg.train.agent['terminal_batch_size'] if terminal_dataset is not None else 0
    batch_size = cfg.train.agent['train_batch_size'] + terminal_batch_size
    
    agent.setup(max_steps=int(num_train_epochs), batch_size=batch_size, ensemble_size=int(cfg.train.vf.ensemble_size))

    total_train_steps = 0

    for epoch_num in pbar:
        #Run validation at some frequency
        if (epoch_num % validation_freq == 0) or (epoch_num == num_train_epochs -1):
            print('Validation...')
            policy.eval(); qf.eval(); vf.eval()
            with record_function('validation'):
                val_metrics, val_figs = agent.compute_validation_metrics(val_dataset, terminal_dataset, normalization_stats)
            policy.train(); qf.train(); vf.train()

            #Log stuff
            row = {}
            for k, v in val_metrics.items():
                row[k.split("/")[-1]] = v
                tb_writer.add_scalar(k, v, total_train_steps)
            tb_writer.add_figure('Validation Ensemble Value Predictions', [v for k,v in val_figs.items()], epoch_num)
            # if cfg.wandb_activate: 
            #     # wandb.log(val_figs, step=step_num)
            #     wandb.log(val_metrics, step=step_num)
            [plt.close(fig) for num, fig in val_figs.items()]
        
        #Rollout Policy at some frequency
        if ((epoch_num + (1-eval_first_iter)) % eval_freq == 0) or (epoch_num == num_train_epochs -1):
            eval_info = {}
            with record_function('rollouts'):
                policy.eval(); qf.eval(); vf.eval()
                if num_eval_episodes > 0:
                    rollout_data, eval_info = evaluate_policy(
                        env, task, mpc_policy, max_episode_steps,
                        num_episodes=num_eval_episodes, 
                        deterministic=True,
                        compute_cost=True,
                        compute_termination=True,
                        discount=cfg.train.agent.discount,
                        normalize_score_fn=lambda returns: normalize_score(cfg.task_name, returns),
                        rng=eval_rng)
                policy.train(); qf.train(); vf.train()

            #Log stuff
            row = {}
            for k, v in eval_info.items():
                row[k.split("/")[-1]] = v
                tb_writer.add_scalar(k, v, total_train_steps)
             # if cfg.wandb_activate: wandb.log(eval_info, step=step_num)

        #Run through training batches
        for batch_num, batch in enumerate(train_dataset.batch_iterator(cfg.train.agent['train_batch_size'])):
            # import pdb; pdb.set_trace()
            batch = dict_to_device(batch, cfg.rl_device)
            full_batch = batch
            if (terminal_dataset is not None) and (terminal_batch_size > 0):
                terminal_batch = terminal_dataset.sample(terminal_batch_size)
                terminal_batch = dict_to_device(terminal_batch, cfg.rl_device)
                full_batch = {k: torch.cat([full_batch[k], terminal_batch[k]], dim=0) for k in full_batch.keys()}
            #Update agent
            with record_function('update'):
                train_metrics = agent.update(
                    full_batch, step_num=total_train_steps, normalization_stats=normalization_stats)
        
            log_metrics = {**log_metrics, **train_metrics, **val_metrics}
            #Log stuff
            row = {}
            for k, v in log_metrics.items():
                row[k.split("/")[-1]] = v
                tb_writer.add_scalar(k, v, total_train_steps)
            total_train_steps += 1


        # if cfg.wandb_activate: wandb.log(log_metrics, step=step_num)
        # logger.row(row)
        save_train = True
        if (epoch_num % cfg.train.agent.checkpoint_freq == 0) or (epoch_num == num_train_epochs -1):
            print(f'Epoch {epoch_num}: Saving current agent to {model_dir}')
            agent_state = agent.state_dict()
            agent_state['normalization_stats'] = normalization_stats
            if save_train:    
                torch.save(agent_state, os.path.join(model_dir, 'agent_checkpoint_50ep_only_rot_goal_0_discount.pt'))
    
        pbar.set_postfix(train_metrics)

    
    if cfg.wandb_activate: wandb.finish()

if __name__ == "__main__":
    main()