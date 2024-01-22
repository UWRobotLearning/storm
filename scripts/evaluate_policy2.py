
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
import os
import hydra
import isaacgym
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
import gym
from storm_kit.learning.policies import MPCPolicy, GaussianPolicy, JointControlWrapper
from storm_kit.learning.replay_buffer import ReplayBuffer
from storm_kit.learning.learning_utils import plot_episode, evaluate_policy
from storm_kit.envs.gym_env_wrapper import GymEnvWrapper
from task_map import task_map


def get_env_and_task(task_name:str, cfg=None): #log max_episode_steps
    if task_name.startswith(('hopper', 'walker2d', 'halfcheetah', 'antmaze')):
        env = gym.make(task_name)
        task = None
        dynamics_model_cls = None
        task_cls=None
    else:
        task_details = task_map[task_name]
        task_cls = task_details['task_cls']    
        dynamics_model_cls = task_details['dynamics_model_cls']
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
                headless=cfg.headless, safe_mode=True
            )
        task = task_cls(
            cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)
        env = GymEnvWrapper(env, task)

    return env, task, task_cls, dynamics_model_cls

@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)    
    base_dir = Path('./tmp_results/{}/{}'.format(cfg.task_name, 'policy_eval'))
    model_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data')
    eval_rng = torch.Generator(device=cfg.rl_device)
    eval_rng.manual_seed(cfg.seed)

    if model_dir is not None:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    if data_dir is not None:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    #Initialize environment
    env, task, task_cls, dyn_model_cls = get_env_and_task(cfg.task_name, cfg=cfg)
    try:
        env.seed(cfg.seed)
    except:
        print('Env does not have seed function')


    #Initialize MPC Policy
    obs_dim = task.obs_dim
    act_dim = task.action_dim
    act_lows, act_highs = task.action_lims
    
    eval_pretrained = cfg.eval.eval_pretrained #and (cfg.eval.pretrained_policy is not None)
    load_pretrained = cfg.eval.load_pretrained #and (cfg.eval.pretrained_policy is not None)
    load_pretrained = eval_pretrained or load_pretrained 

    pretrained_policy = None
    policy_loaded = False
    if eval_pretrained or load_pretrained:
        #load pretrained policy weights
        pretrained_policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, act_lows=act_lows, act_highs=act_highs, device=cfg.rl_device) #task=None,
        checkpoint_path = Path(f'./tmp_results/{cfg.task_name}/BP/models/agent_checkpoint.pt')
        print('Loading agent checkpoint from {}'.format(checkpoint_path))
        try:
            checkpoint = torch.load(checkpoint_path)
            policy_state_dict = checkpoint['policy_state_dict']
            remove_prefix = 'policy.'
            policy_state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in policy_state_dict.items()}
            pretrained_policy.load_state_dict(policy_state_dict)
            pretrained_policy.eval()
            policy_loaded = True
            print('Loaded Pretrained Policy Successfully')
        except:
            policy_loaded = False
            print('Pretrained Policy Not Loaded Successfully')

    if eval_pretrained and policy_loaded:
        print('Evaluating Pretrained Policy')
        policy = pretrained_policy
    else:
        print('Evaluating MPC Policy. Loaded Pretrained?: {}'.format(policy_loaded))
        policy = MPCPolicy(
            obs_dim=obs_dim, act_dim=act_dim, config=cfg.mpc,
            task_cls=task_cls, dynamics_model_cls=dyn_model_cls, 
            sampling_policy=pretrained_policy, device=cfg.rl_device)

    st=time.time()
    num_episodes = cfg.eval.num_episodes
    deterministic_eval = cfg.eval.deterministic_eval
    max_episode_steps = cfg.task.env.get('episodeLength', 1000)
    print('Collecting {0} episodes. Deterministic = {1}, Max Episode Steps = {2}'.format(num_episodes, deterministic_eval, max_episode_steps))

    policy.eval()
    eval_episodes, eval_info = evaluate_policy(
        env, None, policy, max_episode_steps,
        num_episodes=num_episodes, 
        deterministic = deterministic_eval,
        compute_cost=not cfg.real_robot_exp,
        compute_termination=not cfg.real_robot_exp,
        discount=cfg.train.agent.discount,
        normalize_score_fn=None,
        rng=eval_rng)

    print(eval_info)
    buffer = ReplayBuffer(capacity=eval_info['Eval/num_steps'])
    for episode in eval_episodes:
        episode_metrics = task.compute_metrics(episode)
        buffer.add_batch(episode)
        if cfg.debug:
            plot_episode(episode, block=False)
        print(episode_metrics)

    print(buffer)

    print('Time taken = {}'.format(time.time() - st))
    data_dir = data_dir if cfg.eval.save_buffer else None
    # if model_dir is not None:
    #     print('Saving agent to {}'.format(model_dir))
    if data_dir is not None:
        if eval_pretrained: agent_tag = 'pretrained_policy'
        else: agent_tag = 'mpc'
        print('Saving buffer to {}'.format(data_dir))
        buffer.save(os.path.join(data_dir, '{}_buffer_0.pt'.format(agent_tag)))


if __name__ == "__main__":
    main()