from typing import Optional
import torch
from storm_kit.learning.agents import Agent

class MPCAgent(Agent):
    def __init__(
            self, 
            cfg,
            envs,
            task,
            obs_dim,
            action_dim,
            buffer,
            policy,
            runner_fn,
            logger=None,
            tb_writer=None,
            device=torch.device('cpu')):

        super().__init__(
            cfg, envs, task, obs_dim, action_dim,
            buffer=buffer, policy=policy, 
            runner_fn=runner_fn,
            logger=logger, tb_writer=tb_writer,
            device=device,
        )


    def collect_experience(self, num_episodes: int, update_buffer:bool = True):
        
        buff = None
        if update_buffer:
            buff = self.buffer
        
        buff, metrics = self.runner_fn(
            envs = self.envs,
            num_episodes=num_episodes, 
            policy = self.policy,
            task = self.task,
            buffer = buff,
            device=self.device
        )

        if self.logger is not None:
            self.logger.row(metrics)
        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar('Eval/' + k, v, 0)
        return metrics



        