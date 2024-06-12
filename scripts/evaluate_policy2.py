
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
import os
import hydra
import isaacgym
import isaacgymenvs
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
import gym
from storm_kit.learning.policies import MPCPolicy, GaussianPolicy, JointControlWrapper
from storm_kit.learning.value_functions import EnsembleValueFunction
from storm_kit.learning.replay_buffer import ReplayBuffer
from storm_kit.learning.learning_utils import plot_episode, evaluate_policy
from storm_kit.envs.gym_env_wrapper import GymEnvWrapper
from task_map import task_map
import json
import signal
import sys

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
                cfg.task.env, cfg.task.world, cfg.rl_device, 
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
            cfg=cfg.task.task,  world_cfg=cfg.task.world, device=cfg.rl_device, viz_rollouts=False)
        env = GymEnvWrapper(env, task)

    return env, task, task_cls, dynamics_model_cls

def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_tensors(value) for key, value in obj.items()}
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            return obj

def free_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def signal_handler(sig, frame):
    print("Exiting gracefully...")
    free_gpu_memory()
    sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)

@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    free_gpu_memory()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)
    # torch.manual_seed(2)    
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

        model_filename = cfg.eval.vf_trained_agent
        checkpoint_path = Path(f'./tmp_results/{cfg.task_name}/BP/models/{model_filename}')
        checkpoint_path = Path(f'./tmp_results/{cfg.task_name}/BP/models/agent_checkpoint_50ep_ee_acc_twist_obs_discount_0.pt')
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

    #Load pretrained critic
    pretrained_vf = None
    normalization_stats=None
    vf_loaded = False
    if cfg.eval.load_critic:
        #load pretrained critic weights
        pretrained_vf = EnsembleValueFunction(
            obs_dim=obs_dim, config=cfg.train.vf, device=cfg.rl_device)
        model_filename = cfg.eval.vf_trained_agent
        checkpoint_path = Path(f'./tmp_results/{cfg.task_name}/BP/models/{model_filename}')
        # checkpoint_path = Path(f'./tmp_results/{cfg.task_name}/BP/models/agent_checkpoint_50ep_ee_state_obs_ensemble_logsumexp_discount_0.pt')
        print('Loading agent checkpoint from {}'.format(checkpoint_path))
        # pretrained_policy = torch.compile(pretrained_policy)
        try:
            # import pdb; pdb.set_trace()
            checkpoint = torch.load(checkpoint_path)
            vf_state_dict = checkpoint['vf_state_dict']
            remove_prefix = 'vf.'
            vf_state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in vf_state_dict.items()}
            pretrained_vf.load_state_dict(vf_state_dict)
            normalization_stats = checkpoint['normalization_stats']
            pretrained_vf.set_normalization_stats(normalization_stats)
            pretrained_vf.eval()
            vf_loaded = True
            print('Loaded Pretrained VF Successfully')
            pretrained_vf = torch.compile(pretrained_vf)
        except:
            vf_loaded = False
            print('Pretrained VF Not Loaded Successfully')


    if eval_pretrained and policy_loaded:
        print('Evaluating Pretrained Policy')
        policy = pretrained_policy
    else:
        print('Evaluating MPC Policy. Loaded Pretrained?: {}'.format(policy_loaded))
        policy = MPCPolicy(
            obs_dim=obs_dim, act_dim=act_dim, config=cfg.mpc,
            task_cls=task_cls, dynamics_model_cls=dyn_model_cls,
            sampling_policy=pretrained_policy, vf=pretrained_vf if vf_loaded else None, 
            device=cfg.rl_device)
        # policy.set_prediction_metrics(normalization_stats)

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
        compute_cost=True,
        compute_termination=True,
        discount=cfg.train.agent.discount,
        normalize_score_fn=None,
        rng=eval_rng)

    print(eval_info)
    # print(eval_episodes[0])
    # exit()
    buffer_1 = ReplayBuffer(capacity=eval_info['Eval/num_steps'])
    # buffer_2 = ReplayBuffer(capacity=eval_info['Eval/num_steps'])
    # buffer_3 = ReplayBuffer(capacity=eval_info['Eval/num_steps'])
    # buffer_4 = ReplayBuffer(capacity=eval_info['Eval/num_steps'])
    # buffer_5 = ReplayBuffer(capacity=eval_info['Eval/num_steps'])
    # buffer_6 = ReplayBuffer(capacity=eval_info['Eval/num_steps'])
    # buffer_7 = ReplayBuffer(capacity=eval_info['Eval/num_steps'])
    # buffer_8 = ReplayBuffer(capacity=eval_info['Eval/num_steps'])
 
    #more modular
    buffers = [buffer_1]#, buffer_2, buffer_3, buffer_4, buffer_5, buffer_6,buffer_7, buffer_8]
    len_buffer = [50]#,350,300,250,200,150,100,50] #define up to which episode index each buffer should store data
    metrics = []
    # import pdb; pdb.set_trace()
    for index, episode in enumerate(eval_episodes, start=1):
        episode_metrics = task.compute_metrics(episode)
        metrics.append(episode_metrics)
        #add episodes to each buffer based on its permissible length
        for buffer, len in zip(buffers, len_buffer):
            if index <= len:
                buffer.add_batch(episode)
        if cfg.debug:
            plot_episode(episode, block=False)
        print(f"Episode {index}: {episode_metrics}")
    #sanity check for buffer contents
    # import pdb; pdb.set_trace()
    for buf_index, buffer in enumerate(buffers, start=1):
            print(f"Buffer {buf_index}: {buffer}")

    print('Time taken = {}'.format(time.time() - st))
    data_dir = data_dir if cfg.eval.save_buffer else None
    # if model_dir is not None:
    #     print('Saving agent to {}'.format(model_dir))
    # import pdb; pdb.set_trace()
    if data_dir is not None:
        if eval_pretrained: agent_tag = 'pretrained_policy'
        else: agent_tag = 'mpc'
        # print('Saving buffer to {}'.format(data_dir))
        # buffer_1.save(os.path.join(data_dir, '{}_buffer_5ep.pt'.format(agent_tag)))
        # buffer_2.save(os.path.join(data_dir, '{}_buffer_3ep.pt'.format(agent_tag)))
        # buffer_3.save(os.path.join(data_dir, '{}_buffer_1ep.pt'.format(agent_tag)))
        for buffer, length in zip(buffers, len_buffer):
            buffer_filename = os.path.join(data_dir, f'{agent_tag}_buffer_{length}ep_single_cube_center_ee_all_obs_real_robot_jun11_lateral.pt')
            buffer.save(buffer_filename)
            print(f'Saving buffer to {buffer_filename}')
    # print metrics for processing later
    # import pdb; pdb.set_trace()
    print('Buffer saved keys: {}'.format(buffer_1.keys))
    metrics_json = convert_tensors(metrics)
    print(json.dumps(metrics_json))

    tilt_angles = {}
    ee_max_lin_norms = {}
    ee_max_ang_norms = {}
    success={}

    for episode_number, episode_metrics in enumerate(metrics):
        tilt_angles[episode_number] = episode_metrics.get('tilt_angle_max')
        ee_max_lin_norms[episode_number] = episode_metrics.get('ee_lin_vel_twist_max')
        ee_max_ang_norms[episode_number] = episode_metrics.get('ee_ang_vel_twist_max')
        success[episode_number] = episode_metrics.get('success')
    
    print('Tilt Angles: {}'.format(tilt_angles))
    print('EE Max Lin Norms: {}'.format(ee_max_lin_norms))
    print('EE Max Ang Norms: {}'.format(ee_max_ang_norms))
    print('Success: {}'.format(success))

    if KeyboardInterrupt:
        free_gpu_memory()
    

if __name__ == "__main__":
    main()