import copy
from collections import defaultdict
from typing import Optional
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.profiler import record_function

from storm_kit.learning.agents import Agent
from storm_kit.learning.learning_utils import dict_to_device
from storm_kit.mpc.control.control_utils import cost_to_go
import time
from tqdm import tqdm

class BPAgent(Agent):
    def __init__(
        self,
        cfg,
        envs,
        task,
        obs_dim,
        action_dim,
        buffer,
        runner_fn,
        mpc_policy,
        policy=None,
        qf=None,
        vf=None,
        target_qf=None,
        target_vf=None,
        # target_q_func=None,
        # target_critic=None,
        logger=None,
        tb_writer=None,
        device=torch.device('cpu'), 
        eval_rng: Optional[torch.Generator]=None
    ):
        super().__init__(
            cfg, envs, task, obs_dim, action_dim, #obs_space, action_space,
            buffer=buffer, policy=policy,
            runner_fn=runner_fn,
            logger=logger, tb_writer=tb_writer,
            device=device, eval_rng=eval_rng
        )

        self.buffer = self.preprocess_dataset(self.buffer)
        self.mpc_policy = mpc_policy
        self.qf = qf
        self.vf = vf        
        if self.qf is not None:
            assert self.policy is not None, 'Learning a q network requires a policy network.'
            self.target_qf = copy.deepcopy(self.qf).requires_grad_(False) if target_qf is None else target_qf

        if self.vf is not None:
            self.target_vf = copy.deepcopy(self.vf).requires_grad_(False) if target_vf is None else target_vf

        #get full list of parameters
        params = []
        for net in (self.policy, self.qf, self.vf):
            if net is not None:
                params += list(net.parameters())

        self.optimizer = optim.Adam(params, lr=self.cfg['optimizer']['lr'])

        # if self.policy is not None:
        #     self.policy_optimizer = optim.Adam(self.policy.parameters(), 
        #                                 lr=self.cfg['policy_optimizer']['lr'])
        # if self.critic is not None:
        #     self.critic_optimizer = optim.Adam(self.critic.parameters(), 
        #                                 lr=self.cfg['critic_optimizer']['lr'])
 
        self.policy_loss_type = self.cfg['policy_loss_type']
        self.num_action_samples = self.cfg['num_action_samples']
        # self.fixed_alpha = self.cfg['fixed_alpha']
        if self.policy_loss_type not in ["mse", "nll"]:
            raise ValueError('Unidentified policy loss type {}.'.format(self.policy_loss_type))
        self.num_eval_episodes = self.cfg.get('num_eval_episodes', 1)
        self.eval_first_policy = self.cfg.get('eval_first_policy', False)
        self.policy_use_tanh = self.cfg.get('policy_use_tanh', False)
        self.discount = self.cfg.get('discount')
        self.polyak_tau = float(self.cfg['polyak_tau'])
        # self.best_policy = copy.deepcopy(self.policy)

    def train(self, model_dir=None, data_dir=None, debug:bool=False):
        num_train_steps = self.cfg['num_pretrain_steps']
        log_metrics = defaultdict(list)
        pbar = tqdm(range(int(num_train_steps)), desc='train')

        for i in pbar:
            #Evaluate policy at some frequency
            if ((i + (1-self.eval_first_policy)) % self.eval_freq == 0) or (i == num_train_steps -1):
                print('[BehaviorPretraining]: Evaluating policy')
                self.policy.eval()
                eval_buffer, eval_metrics = self.evaluate_policy(
                    self.mpc_policy, 
                    num_eval_episodes=self.num_eval_episodes, 
                    deterministic=True, 
                    debug=False)
               
                print(eval_metrics)
                for k,v in eval_metrics.items():
                    log_metrics['eval/episode/{}'.format(k)].append(v)
                episode_metric_list = [self.task.compute_metrics(episode) for episode in eval_buffer.episode_iterator(
                    max_episode_length=self.envs.max_episode_length - 1)]
                episode_metrics = defaultdict(list)
                for k in episode_metric_list[0].keys():
                    [episode_metrics[k].append(l[k]) for l in episode_metric_list]
                for k,v in episode_metrics.items():
                    log_metrics['eval/episode/{}'.format(k)].extend(v)

                self.policy.train()
                pbar.set_postfix(eval_metrics)

            with record_function('sample_batch'):
                batch = self.buffer.sample(self.cfg['train_batch_size']) #, sample_next_state=False)
                batch = dict_to_device(batch, self.device)
            
            if self.relabel_data:
                with record_function('relabel_data'):
                    batch = self.preprocess_batch(batch, compute_cost_and_terminals=True)
            
            with record_function('update'):
                train_metrics = self.update(batch, i)
            pbar.set_postfix(train_metrics)

            for k,v in train_metrics.items():
                log_metrics['train/losses/{}'.format(k)].append(v)

            #Log stuff
            row = {}
            for k, v in log_metrics.items():
                row[k.split("/")[-1]] = v[-1]
                if self.tb_writer is not None:                        
                    self.tb_writer.add_scalar(k, v[-1], i)
            if self.logger is not None:
                self.logger.row(row)

            # if self.tb_writer is not None:
            #     for k, v in train_metrics.items():
            #         self.tb_writer.add_scalar('Train/' + k, v, i)
                        
            if (i % self.checkpoint_freq == 0) or (i == num_train_steps -1):
                print(f'Iter {i}: Saving current policy')
                self.save(model_dir, data_dir, iter=0)
            

    def update(self, batch_dict, step_num):

        policy_info_dict, qf_info_dict = {}, {}
        total_loss = torch.tensor(0., device=self.device)
        if self.policy is not None:
            policy_loss, policy_info_dict = self.compute_policy_loss(batch_dict)
            total_loss += policy_loss

        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # self.policy_optimizer.step()

        if self.qf is not None:
            qf_loss, qf_info_dict = self.compute_critic_loss(batch_dict)
            total_loss += qf_loss
            # self.critic_optimizer.zero_grad()
            # qf_loss.backward()
            # self.critic_optimizer.step()


        #TODO: Add value function learning
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # for p1, p2 in zip(self.mpc_policy.policy.controller.sampling_policy.parameters(), self.policy.parameters()):
        #     assert torch.allclose(p2, p2)
        # print([p for p in self.mpc_policy.policy.controller.sampling_policy.parameters()])
        # input('...')
        # for p1, p2 in zip(self.mpc_policy.policy.controller.value_function.parameters(), self.qf.parameters()):
        #     assert torch.allclose(p1, p2)
        # print([p for p in self.mpc_policy.policy.controller.value_function.parameters()])
        # input('...')


        info_dict = {**policy_info_dict, **qf_info_dict}
        
        return info_dict


    def compute_policy_loss(self, batch_dict):
        obs_batch = batch_dict['obs']
        state_batch = batch_dict['state_dict']
        act_batch = batch_dict['actions']
        
        if self.policy_use_tanh:
            act_batch = torch.tanh(act_batch)

        if self.policy_loss_type == 'mse':
            new_actions = self.policy.get_action({'obs':obs_batch}, deterministic=False, num_samples=self.num_action_samples)            
            if new_actions.dim() == 3:
                act_batch = act_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)
            policy_imitation_loss = F.mse_loss(new_actions, act_batch, reduction='none')
            policy_imitation_loss = policy_imitation_loss.sum(-1).mean()
            policy_loss = policy_imitation_loss
        
        elif self.policy_loss_type == 'nll':
            policy_input = {
                'states': state_batch,
                'obs': obs_batch}
            dist = self.policy(policy_input)
            policy_loss = -1.0 * dist.log_prob(act_batch).mean()
            new_actions = dist.sample()
            log_pi_new_actions = dist.log_prob(new_actions).mean()
            action_diff = torch.norm(new_actions - act_batch, dim=-1).mean()
            # policy_loss = -1.0 * self.policy.log_prob(policy_input, act_batch).mean()

            # action_dist = self.policy({'obs':obs_batch})
            # policy_imitation_loss = -1.0 * action_dist.log_prob(act_batch).mean()
            # #compute policy entropy
            # new_actions = action_dist.rsample()
            # log_prob_new_actions = action_dist.log_prob(new_actions)
            # policy_entropy = -log_prob_new_actions.mean()
            # policy_loss = policy_imitation_loss - self.fixed_alpha * policy_entropy

        # new_actions, log_pi_new_actions = self.policy.entropy(policy_input)
        
        policy_info_dict = {
            'policy_loss': policy_loss.item(),
            'action_difference': action_diff.item(),
            'policy_entropy': log_pi_new_actions.item()
        }

        return policy_loss, policy_info_dict
    
    def compute_critic_loss(self, batch_dict):
        assert self.policy is not None, "Q-function learning requires policy"
        cost_batch = batch_dict['cost']
        obs_batch = batch_dict['obs']
        act_batch = batch_dict['actions']
        next_obs_batch = batch_dict['next_obs']
        next_state_batch = batch_dict['next_state_dict']
        done_batch = batch_dict['terminals'].float()

        #Update target critic using exponential moving average
        for (param1, param2) in zip(self.target_qf.parameters(), self.qf.parameters()):
            param1.data.mul_(1. - self.polyak_tau).add_(param2.data, alpha=self.polyak_tau)

        with torch.no_grad():
            policy_input = {
                'states': next_state_batch,
                'obs': next_obs_batch}
            
            next_actions_dist = self.policy(policy_input)                
            next_actions = next_actions_dist.sample(torch.Size([self.num_action_samples]))
            # next_actions_log_prob = next_actions_dist.log_prob(next_actions).mean()

            q_pred_next = self.target_qf(
                {'obs': next_obs_batch.unsqueeze(0).repeat(self.num_action_samples, 1, 1)}, 
                next_actions).mean(0)
            
            q_target = cost_batch +  (1. - done_batch) * self.discount * q_pred_next

        qf_all = self.qf.all({'obs': obs_batch}, act_batch)
        q_target = q_target.unsqueeze(-1).repeat(1, qf_all.shape[-1])

        qf_loss = F.mse_loss(qf_all, q_target, reduction='none')
        qf_loss = qf_loss.sum(-1).mean(0) #sum along ensemble dimension and mean along batch

        avg_q_value = torch.max(qf_all, dim=-1)[0].mean() 
        avg_target_value = q_target.mean()
        max_target_value = q_target.max()#max instead of min since we are minimizing costs
       
        qf_info_dict = {
            'qf_loss': qf_loss.item(),
            'avg_q_value': avg_q_value.item(),
            'avg_target_value': avg_target_value.item(),
            'max_target_value': max_target_value.item()
        }
        return qf_loss, qf_info_dict


    def preprocess_dataset(self, buffer):

        for episode in buffer.episode_iterator():
            H = len(episode['actions'])
            

        buffer = buffer.qlearning_dataset()
        return buffer

