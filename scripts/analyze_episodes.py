
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import hydra
import torch
from storm_kit.learning.replay_buffers import RobotBuffer
from storm_kit.learning.learning_utils import buffer_dict_from_folder
from task_map import task_map
import matplotlib.pyplot as plt



@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)

    task_details = task_map[cfg.task.name]
    task_cls = task_details['task_cls']   
    
    base_dir = Path('./tmp_results/{}/{}'.format(cfg.task_name, 'policy_eval'))
    model_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data')
    data_dir = os.path.join(data_dir, 'mpc') #TODO: Generalize

    #Initialize task
    task = task_cls(
        cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)

    #Load bufffers from folder
    buffer_dict = buffer_dict_from_folder(data_dir)

    for episode, episode_data in buffer_dict.items():
        print(episode_data['goal_dict'])
        plt.plot(episode_data['state_dict']['q_pos'])
        input('...')

    print(episode_data)



if __name__ == "__main__":
    main()