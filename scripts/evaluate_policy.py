
from pathlib import Path
import isaacgym 
from omegaconf import DictConfig
import numpy as np
import os
import hydra
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
from storm_kit.learning.policies import MPCPolicy, GaussianPolicy, JointControlWrapper
from storm_kit.learning.learning_utils import episode_runner, plot_episode
from task_map import task_map

@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)

    task_details = task_map[cfg.task.name]
    task_cls = task_details['task_cls']   
    
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

    #Initialize MPC Policy
    obs_dim = task.obs_dim
    act_dim = task.action_dim
    act_lows, act_highs = task.action_lims
    state_bounds = task.state_bounds

    # buffer = RobotBuffer(capacity=int(1e6), device=cfg.rl_device)  
    
    eval_pretrained = cfg.eval.eval_pretrained and (cfg.eval.pretrained_policy is not None)
    load_pretrained = cfg.eval.load_pretrained and (cfg.eval.pretrained_policy is not None)
    load_pretrained = eval_pretrained or load_pretrained 

    pretrained_policy = None
    policy_loaded = False
    if eval_pretrained or load_pretrained:
        #we provide a task to policy as well to re-calculate observations
        policy_task = task_cls(
            cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)
        #load pretrained policy weights
        pretrained_policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, task=policy_task, act_lows=act_lows, act_highs=act_highs, device=cfg.rl_device)
        checkpoint_path = Path('./tmp_results/{}/{}'.format(cfg.task_name, cfg.eval.pretrained_policy))
        print('Loading policy from {}'.format(checkpoint_path))
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
        dyn_model_cls = task_details['dynamics_model_cls']
        policy = MPCPolicy(
            obs_dim=obs_dim, act_dim=act_dim, config=cfg.mpc,
            task_cls=task_cls, dynamics_model_cls=dyn_model_cls, 
            sampling_policy=pretrained_policy, device=cfg.rl_device)
        policy_task = None

    policy = JointControlWrapper(
        config=cfg.task.joint_control, 
        policy=policy, 
        state_bounds=state_bounds,
        device=cfg.rl_device)

    st=time.time()
    num_episodes = cfg.eval.num_episodes
    deterministic_eval = cfg.eval.deterministic_eval
    print('Collecting {} episodes. Deterministic = {}'.format(num_episodes, deterministic_eval))
    buffer, metrics = episode_runner(
        envs,
        num_episodes = num_episodes, 
        policy = policy,
        task = task,
        collect_data = True,
        deterministic = deterministic_eval,
        debug = False,
        check_termination= not cfg.real_robot_exp,
        device = cfg.rl_device,
        rng = eval_rng)
    print(metrics)

    for episode in buffer.episode_iterator():
        episode_metrics = task.compute_metrics(episode)
        if cfg.debug:
            plot_episode(episode, block=False)
        print(episode_metrics)

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