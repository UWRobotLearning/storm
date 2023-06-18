
from pathlib import Path
import isaacgym 
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import hydra
import torch
torch.manual_seed(0)
from torch.profiler import profile, record_function, ProfilerActivity
import time


from storm_kit.rl.policies import MPCPolicy
from storm_kit.rl.agents import MPCAgent
from storm_kit.rl.replay_buffer import ReplayBuffer


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    import isaacgymenvs
    import storm_kit.envs

    envs = isaacgymenvs.make(
        cfg.seed, 
        cfg.task_name, 
        cfg.task.env.numEnvs, 
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
        # **kwargs,
    )

    base_dir = Path('./tmp_results/{}/{}'.format(cfg.task_name, cfg.train.agent.name))
    model_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    cfg.mpc.env_control_space = cfg.task.env.controlSpace
    cfg.mpc.world = cfg.task.world
    cfg.mpc.control_dt = cfg.task.sim.dt

    obs_dim = envs.obs_space.shape[0]
    act_dim = envs.action_space.shape[0]
    
    buffer = ReplayBuffer(capacity=int(1e6), obs_dim=obs_dim, act_dim=act_dim, device=cfg.rl_device)    
    policy = MPCPolicy(obs_dim=obs_dim, act_dim=act_dim, config=cfg.mpc, device=cfg.rl_device)
    agent = MPCAgent(cfg.train.agent, envs=envs, obs_space=envs.obs_space, action_space=envs.action_space, 
                     buffer=buffer, policy=policy, device=cfg.rl_device)
    st=time.time()
    agent.collect_experience(num_steps_per_env=cfg.train.agent.num_steps_per_env, update_buffer=True)
    print('Time taken = {}'.format(time.time() - st))
    print('Saving agent. Buffer save = {}'.format(cfg.train.agent.save_buffer))
    agent.save(model_dir, save_buffer=cfg.train.agent.save_buffer)
    print('Agent saved')


    # obs = envs.reset()
    # actor.update_goal(obs['goal'])
    # # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    # st1=time.time()
    # total_mpc_time = 0.0
    # for i in range(1000):
    #     with record_function("mpc_inference"):
    #         # st=time.time()
    #         command = actor.forward(obs)   
    #         # mpc_time = time.time()-st
    #         # total_mpc_time += mpc_time
        
    #     # if i == 1:
    #     #     exit()
    #     # command = actor.forward(obs)

    #     # q_des = (np.pi/4.) * torch.ones([20, 7], device=torch.device('cuda')) 
    #     obs, reward, done, info = envs.step(command['qd_des'])
    #     # print('env', reward.item())
    #     # action = envs.zero_actions()
    #     # obs, reward, done, info = envs.step(action)

    #     envs.render()
    # print('total time = ', total_mpc_time)
    # print(prof.key_averages().table(sort_by = "cuda_time_total", row_limit=500))


if __name__ == "__main__":
    main()