
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import hydra
import torch
from storm_kit.learning.replay_buffer import ReplayBuffer
from storm_kit.learning.learning_utils import buffer_dict_from_folder, plot_episode, preprocess_dataset
from task_map import task_map
import seaborn as sns
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

    #Initialize task
    task = task_cls(
        cfg=cfg.task.task, device=cfg.rl_device, viz_rollouts=False, world_params=cfg.task.world)

    #Load bufffers from folder
    buffer_dict = buffer_dict_from_folder(data_dir)


    for buffer_name, buffer in buffer_dict.items():
        print('Analyzing buffer {}'.format(buffer_name))
        buffer, validation_buffer, success_buffer, buffer_info = preprocess_dataset(buffer, None, task, cfg.train.agent)

        ee_goal = buffer['goals/ee_goal'] if 'goals/ee_goal' in buffer else None
        if ee_goal is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(ee_goal[:,0], ee_goal[:,1], ee_goal[:,2], c=ee_goal[:,2], cmap='viridis', marker='o')
            ax.set_xlabel('X(m)')
            ax.set_ylabel('Y(m)')
            ax.set_zlabel('Z(m)')
            ax.set_title('Goal positions')




        for episode in buffer.episode_iterator():
            episode_metrics = task.compute_metrics(episode)
            plot_episode(episode, block=False)
            print(episode_metrics)




if __name__ == "__main__":
    main()