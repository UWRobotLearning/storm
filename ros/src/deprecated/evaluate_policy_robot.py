
import rospy
from pathlib import Path
import isaacgym 
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import hydra
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
from storm_kit.learning.policies import MPCPolicy, GaussianPolicy, JointControlWrapper
from storm_kit.tasks import ArmReacher
from storm_kit.learning.replay_buffers import RobotBuffer
from storm_kit.learning.learning_utils import episode_runner, minimal_episode_runner

task_map = {}
task_map['FrankaReacherRealRobot'] = {
    'task_cls': ArmReacher
}
task_map['FrankaTrayReacherRealRobot'] = {
    'task_cls': ArmReacher
}


@hydra.main(config_name="config", config_path="../../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)

    torch.set_default_dtype(torch.float32)
    from franka_real_robot_env import FrankaRealRobotEnv

    rospy.init_node("eval_policy_robot", anonymous=True, disable_signals=False)    

    task_details = task_map[cfg.task.name]
    task_cls = task_details['task_cls']   
    
    base_dir = Path('../tmp_results/{}/{}'.format(cfg.task_name, 'policy_eval'))
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
    envs = FrankaRealRobotEnv(
        cfg.task, 
        cfg.rl_device)

    #Initialize task
    task = task_cls(
        cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)

    #Initialize MPC Policy
    obs_dim = task.obs_dim
    act_dim = task.action_dim
    act_lows, act_highs = task.action_lims
    
    eval_pretrained = cfg.eval.eval_pretrained and (cfg.eval.pretrained_policy is not None)

    if eval_pretrained:
        #we provide a task to policy as well to re-calculate observations
        policy_task = task_cls(
            cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)
        #load pretrained policy weights
        policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.train.policy, task=policy_task, act_lows=act_lows, act_highs=act_highs, device=cfg.rl_device)
        checkpoint_path = Path('../tmp_results/{}/{}'.format(cfg.task_name, cfg.eval.pretrained_policy))
        print('Loading policy from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        policy_state_dict = checkpoint['policy_state_dict']
        remove_prefix = 'policy.'
        policy_state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in policy_state_dict.items()}
        policy.load_state_dict(policy_state_dict)
        policy.eval()

    else:
        print('Evaluating MPC Policy')
        policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.mpc, task_cls=task_cls, device=cfg.rl_device)
        policy_task = None

    policy = JointControlWrapper(config=cfg.task.joint_control, policy=policy, device=cfg.rl_device)
    # #Initialize Agent
    # agent = MPCAgent(cfg.eval, envs=envs, task=task, obs_dim=obs_dim, action_dim=act_dim, 
    #                  buffer=buffer, policy=policy, runner_fn=episode_runner, device=cfg.rl_device)
    st=time.time()
    num_episodes = cfg.eval.num_episodes
    deterministic_eval = cfg.eval.deterministic_eval
    print('Collecting {} episodes. Deterministic = {}'.format(num_episodes, deterministic_eval))

    episode_buffer_list = []
    data_dir = data_dir if cfg.eval.save_buffer else None

    for ep_num in range(num_episodes):
        episode_buffer = RobotBuffer(capacity=int(envs.max_episode_length), device=cfg.rl_device)  
        metrics = minimal_episode_runner(
            envs,
            num_episodes = 1, 
            policy = policy,
            task = task,
            buffer = episode_buffer,
            deterministic = deterministic_eval,
            debug = False,
            device = cfg.rl_device,
            rng = eval_rng)
        
        print('Collected episode {}'.format(ep_num))
        # if model_dir is not None:
        #     print('Saving agent to {}'.format(model_dir))
        if data_dir is not None:
            if eval_pretrained: agent_tag = 'pretrained_policy'
            else: agent_tag = 'mpc'
            #TODO: Here add option to not overwrite but save data incrementally
            buffer_dir = os.path.join(data_dir, '{}'.format(agent_tag))
            if not os.path.exists(buffer_dir): os.makedirs(buffer_dir)
            buffer_file = os.path.abspath(os.path.join(buffer_dir, 'episode_{}.pt'.format(ep_num)))
            print('Saving episode buffer to {}. Buffer len: {}'.format(buffer_file, len(episode_buffer)))
            episode_buffer.save(buffer_file)
    
    print('Total time taken = {}'.format(time.time() - st))


if __name__ == "__main__":
    main()