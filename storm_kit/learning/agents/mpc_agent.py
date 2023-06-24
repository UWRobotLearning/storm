from typing import Optional
import torch
from storm_kit.rl.agents import Agent

class MPCAgent(Agent):
    def __init__(
            self, 
            cfg,
            envs,
            obs_space,
            action_space,
            buffer,
            policy,
            logger=None,
            tb_writer=None,
            device=torch.device('cpu')):

        super().__init__(
            cfg, envs, obs_space, action_space,
            buffer=buffer, policy=policy, 
            logger=logger, tb_writer=tb_writer,
            device=device,
        )

    def collect_experience(self, num_steps_per_env: int, update_buffer:bool = True):
        metrics = super().collect_experience(num_steps_per_env, update_buffer)
        if self.logger is not None:
            self.logger.row(metrics)
        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar('Eval/' + k, v, 0)
        return metrics



        