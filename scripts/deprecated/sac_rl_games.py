import isaacgym 
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import hydra
import torch
torch.manual_seed(0)

from torch.profiler import profile, record_function, ProfilerActivity
import time
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env



class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        mean = self.net(inputs["obs"])
        return mean, self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["obs"], inputs["taken_actions"]], dim=1)), {}

# @hydra.main(config_name="config", config_path="../content/configs/gym")
def main(): #cfg: DictConfig
    import isaacgymenvs
    # from storm_kit.gym import tasks
    torch.set_default_dtype(torch.float32)
    seed=42
    envs = isaacgymenvs.make(
        seed=seed,
        task="AllegroHand",
        num_envs=100,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=False)


#     envs = isaacgymenvs.make(
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

    envs = wrap_env(envs)
    device = envs.device

    memory = RandomMemory(memory_size=20000, num_envs=envs.num_envs, device=device, replacement=True)

    models_sac = {}
    models_sac["policy"] = StochasticActor(envs.observation_space, envs.action_space, device, clip_actions=False)
    models_sac["critic_1"] = Critic(envs.observation_space, envs.action_space, device)
    models_sac["critic_2"] = Critic(envs.observation_space, envs.action_space, device)
    models_sac["target_critic_1"] = Critic(envs.observation_space, envs.action_space, device)
    models_sac["target_critic_2"] = Critic(envs.observation_space, envs.action_space, device)


    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for model in models_sac.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    
    cfg_sac = SAC_DEFAULT_CONFIG.copy()
    cfg_sac["gradient_steps"] = 1
    cfg_sac["batch_size"] = 512
    cfg_sac["random_timesteps"] = 0
    cfg_sac["learning_starts"] = 0
    cfg_sac["learn_entropy"] = True
    # logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
    cfg_sac["experiment"]["write_interval"] = 25
    cfg_sac["experiment"]["checkpoint_interval"] = 1000

    agent_sac = SAC(models=models_sac,
                    memory=memory,
                    cfg=cfg_sac,
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    device=device)
    
    # Configure and instantiate the RL trainer
    cfg_train = {"timesteps": 1000000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_train,
                                env=envs,
                                agents=[agent_sac],
                                agents_scope=[])

    # start training
    trainer.train()




if __name__ == "__main__":
    main()