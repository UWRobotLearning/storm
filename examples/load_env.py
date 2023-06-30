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

    envs.render()
    envs.step(torch.zeros(3))
    input('....')

    obs_dim = envs.obs_space.shape[0]
    act_dim = envs.action_space.shape[0]
    


if __name__ == "__main__":
    main()