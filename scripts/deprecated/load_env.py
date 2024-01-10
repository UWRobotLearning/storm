from pathlib import Path
import os
import isaacgym 
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from datetime import datetime


@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    import isaacgymenvs
    # import storm_kit.envs
    from storm_kit.envs import FrankaEnv, PointRobotEnv
    torch.set_default_dtype(torch.float32)

    # envs = isaacgymenvs.make(
    #     cfg.seed, 
    #     cfg.task_name, 
    #     cfg.task.env.numEnvs, 
    #     cfg.sim_device,
    #     cfg.rl_device,
    #     cfg.graphics_device_id,
    #     cfg.headless,
    #     cfg.multi_gpu,
    #     cfg.capture_video,
    #     cfg.force_render,
    #     cfg,
    #     # **kwargs,
    # )

    if cfg.task.name in ["FrankaReacher", "FrankaPusher"]:
        envs = FrankaEnv(
            cfg.task, 
            torch.device(cfg.rl_device), 
            torch.device(cfg.sim_device), 
            cfg.graphics_device_id, 
            cfg.headless, 
            False, 
            cfg.force_render
        )
    else:
        envs = PointRobotEnv(
            cfg.task, 
            torch.device(cfg.rl_device), 
            torch.device(cfg.sim_device), 
            cfg.graphics_device_id, 
            cfg.headless, 
            False, 
            cfg.force_render
        )

    envs.render()
    while True:
        state_dict = envs.step({
        'q_pos_des': torch.zeros(1),
        'q_vel_des': torch.zeros(1),
        'q_acc_des': torch.zeros(1),
        'effort': torch.tensor([0.5, 0.0])})
        print(state_dict)

if __name__ == "__main__":
    main()