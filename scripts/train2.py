import copy
from collections import defaultdict
from pathlib import Path
from typing import Union, Tuple
import hydra
from omegaconf import DictConfig
import time 
import gym
import d4rl
import numpy as np
import isaacgym
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
                cfg.task, cfg.rl_device, 
                cfg.sim_device, cfg.graphics_device_id, 
                cfg.headless, False, cfg.force_render
            )
        else:
            from storm_kit.envs.panda_real_robot_env import PandaRealRobotEnv
            env = PandaRealRobotEnv(
                cfg.task, device=cfg.rl_device,
                headless=cfg.headless, safe_mode=False
            )
        task = task_cls(
            cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)
        #Load dataset
        base_dir = Path('./tmp_results/{}/{}'.format(cfg.task_name, 'policy_eval'))
        model_dir = os.path.join(base_dir, 'models')
        data_dir = os.path.join(base_dir, 'data')

        #Initialize task
        task = task_cls(
            cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)

        #Load bufffers from folder
        buffer_dict = buffer_dict_from_folder(data_dir)
        replay_buffer = buffer_dict['mpc_buffer_0.pt']

    return env, task, replay_buffer

def get_mpc_policy(cfg, policy=None, vf=None, qf=None, V_min=-float('inf'), V_max=float('inf')
):
    task_name = cfg.task_name
    task_details = task_map[task_name]
    task_cls = task_details['task_cls']    
    dynamics_model_cls = task_details['dynamics_model_cls']

    task = task_cls(
        cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)

    obs_dim = task.obs_dim
    act_dim = task.action_dim

    mpc_policy = MPCPolicy(
        obs_dim=obs_dim, act_dim=act_dim, config=cfg.mpc,
        task_cls=task_cls, dynamics_model_cls=dynamics_model_cls, 
        sampling_policy=policy, vf=vf, qf=qf, V_min=V_min, V_max=V_max,
        device=cfg.rl_device)
    del task
    return mpc_policy


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean.cpu().numpy(), std.cpu().numpy()


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)

    env, task, dataset = get_task_and_dataset(cfg.task_name, cfg)
    dataset, dataset_info = preprocess_dataset(
        dataset, env, task, cfg.train.agent, 
        normalize_score_fn=lambda returns: normalize_score(cfg.task_name, returns))
    dataset = qlearning_dataset(dataset=dataset)
    print(dataset_info)

    if cfg.train.agent.normalize_states:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1
    # dataset["observations"] = normalize_states(
    #     dataset["observations"], state_mean, state_std
    # )
    # dataset["next_observations"] = normalize_states(
    #     dataset["next_observations"], state_mean, state_std
    # )
    # env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    env = GymEnvWrapper(env, task)

    logging_info = init_logging(cfg.task_name, cfg.train.agent.name, cfg)
    model_dir = logging_info['model_dir']
    logger = logging_info['log']
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
    policy = GaussianPolicy(
        obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, task=task, device=cfg.rl_device)
    # qf = TwinQFunction(
    #     obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.qf, device=cfg.rl_device)
    qf = EnsembleQFunction(
        obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.qf, device=cfg.rl_device)
    target_qf = copy.deepcopy(qf).requires_grad_(False)
    vf = EnsembleValueFunction(
        obs_dim=obs_dim, config=cfg.train.vf, device=cfg.rl_device)
    target_vf = copy.deepcopy(vf).requires_grad_(False)

    #initialize mpc policy
    V_min, V_max = dataset_info['V_min'], dataset_info['V_max']
    mpc_policy = get_mpc_policy(cfg, policy=None, vf=vf, qf=qf, V_min=V_min, V_max=V_max)

    # Run behavior pretraining
    num_train_steps = int(cfg.train.agent['num_pretrain_steps'])
    agent = BPAgent(
        cfg.train.agent, env, task, obs_dim, act_dim,
        dataset, runner_fn=None, policy=None, qf=None, vf=vf,
        target_qf=None, target_vf=target_vf,
        max_steps=num_train_steps, V_min=V_min, V_max=V_max, 
        logger=None, tb_writer=None,
        device=cfg.rl_device, eval_rng=eval_rng
    )

    log_metrics = defaultdict(list)
    pbar = tqdm(range(int(num_train_steps)), desc='train')
    num_eval_episodes = cfg.train.agent.num_eval_episodes
    eval_first_policy = cfg.train.agent.eval_first_policy
    eval_freq = cfg.train.agent.eval_freq
    for step_num in pbar:
        #Evaluate policy at some frequency
        eval_info = {}
        if ((step_num + (1-eval_first_policy)) % eval_freq == 0) or (step_num == num_train_steps -1):
            print('Evaluating policy')
            policy.eval()
            eval_data, eval_info = evaluate_policy(
                env, task, mpc_policy, 1000,
                num_episodes=num_eval_episodes, 
                deterministic=True,
                check_termination=True,
                discount=cfg.train.agent.discount,
                normalize_score_fn=lambda returns: normalize_score(cfg.task_name, returns),
                rng=eval_rng)
            print(eval_info)
            #Log stuff
            row = {}
            for k, v in eval_info.items():
                row[k.split("/")[-1]] = v
                tb_writer.add_scalar(k, v, step_num)
            policy.train()
        #Sample batch of data
        with record_function('sample_batch'):
            batch = dataset.sample(cfg.train.agent['train_batch_size'])
            batch = dict_to_device(batch, cfg.rl_device)
        #Update agent
        with record_function('update'):
            train_metrics = agent.update(batch_dict=batch, step_num=step_num)
            # pbar.set_postfix(train_metrics)
        
        log_metrics = {**log_metrics, **train_metrics}
        #Log stuff
        row = {}
        for k, v in log_metrics.items():
            row[k.split("/")[-1]] = v
            tb_writer.add_scalar(k, v, step_num)
        # logger.row(row)

        if (step_num % cfg.train.agent.checkpoint_freq == 0) or (step_num == num_train_steps -1):
            print(f'Iter {step_num}: Saving current policy')
            agent.save(model_dir, None, iter=0)
            
if __name__ == "__main__":
    main()