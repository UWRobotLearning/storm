from collections import defaultdict
import csv, os, sys
import copy
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from typing import Optional
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchaudio
from scipy import signal
from storm_kit.learning.replay_buffer import ReplayBuffer, train_val_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.profiler import profile, record_function, ProfilerActivity


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    # rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}' #_{rand_str}'

def logmeanexp(x, dim):
    max_x, _ = torch.max(x, dim=dim, keepdim=True)
    return torch.squeeze(max_x, dim=dim) + torch.log(torch.mean(torch.exp((x - max_x)), dim=dim))

def dict_to_device(d, device):
    for k, v in d.items():
        if isinstance(v, dict):
            for k1 in v.keys():
                d[k][k1] = d[k][k1].to(device)
        else:
            d[k] = d[k].to(device)
    return d

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.yaml',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        # (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        OmegaConf.save(cfg_dict, self.dir/cfg_filename)            
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n', nostdout=False):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        buff_list = [self.txt_file]
        if not nostdout:
            buff_list += [sys.stdout]
        for f in buff_list:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict, nostdout=False):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict), nostdout=nostdout)
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()

def buffer_from_file(filepath):
    print('Loading buffer from {}'.format(filepath))
    buffer = ReplayBuffer(capacity=1000)
    buffer.load(filepath)
    print('Loaded buffer {}'.format(buffer))
    return buffer, len(buffer)

def single_buffer_from_folder(data_dir, capacity=None, device=torch.device('cpu')):
    files = sorted(os.listdir(data_dir))
    #TODO: Sort by episode number!!!!
    episode_buffers = []
    total_data_points = 0
    for file in files:
        filepath = os.path.join(data_dir, file)
        #TODO: ensure buffers have consistent dimensions for obs, act 
        ep_buff, num_datapoints = buffer_from_file(filepath)
        episode_buffers.append(ep_buff) 
        total_data_points += num_datapoints
    
    # obs_dim = episode_buffers[0].obs_dim
    # act_dim = episode_buffers[0].act_dim
    # buffer = ReplayBuffer(capacity=total_data_points, obs_dim=obs_dim, act_dim=act_dim, device=device)
    buffer_capacity=total_data_points if capacity is None else capacity
    buffer = ReplayBuffer(buffer_capacity, 7, device)
    for b in episode_buffers:
        buffer.concatenate(b.state_dict()) 
    return buffer

def buffer_dict_from_folder(data_dir, capacity=None, device=torch.device('cpu')):
    buffer_dict = {}
    files = sorted(os.listdir(data_dir))
    total_data_points = 0
    for file in files:
        filepath = os.path.join(data_dir, file)
        #TODO: ensure buffers have consistent dimensions for obs, act 
        ep_buff, num_datapoints = buffer_from_file(filepath)
        buffer_dict[file] = ep_buff 
        total_data_points += num_datapoints
    return buffer_dict


def fit_mlp(
    net:torch.nn.Module, 
    x_train:torch.Tensor, y_train:torch.Tensor,
    x_val:torch.Tensor, y_val:torch.Tensor, 
    optimizer, loss_fn, 
    num_epochs:int, batch_size:int, 
    x_train_aux=None, y_train_aux=None,
    x_val_aux=None, y_val_aux=None,
    aux_batch_size=1, normalize=False, 
    is_classifier=False, device:torch.device=torch.device('cpu')):

    norm_dict = None
    if normalize:
        norm_dict = {}
        mean_x = torch.mean(x_train, dim=0)
        std_x = torch.std(x_train, dim=0)
        mean_y = torch.mean(y_train, dim=0)
        std_y = torch.std(y_train, dim=0)

        x_train = torch.div((x_train - mean_x), std_x + 1e-6)
        x_val = torch.div((x_val - mean_x), std_x + 1e-6)
        
        if not is_classifier:
            #normalize the targets
            y_train = torch.div((y_train - mean_y), std_y + 1e-6)
            y_val = torch.div((y_val - mean_y), std_y + 1e-6)


        norm_dict['x'] = {'mean':mean_x, 'std':std_x}
        norm_dict['y'] = {'mean':mean_y, 'std':std_y}


    train_dataset = TensorDataset(x_train, y_train)
    # val_dataset = TensorDataset(x_val, y_val)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    auxtrainloader = None
    if x_train_aux is not None:
        if normalize:
            x_train_aux = torch.div((x_train_aux - mean_x), std_x + 1e-6)
            if not is_classifier:
                y_train_aux = torch.div((y_train_aux - mean_y), std_y + 1e-6)

        aux_train_dataset = TensorDataset(x_train_aux, y_train_aux)
        auxtrainloader = DataLoader(aux_train_dataset, batch_size=aux_batch_size, shuffle=True)

    net.to(device)

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    pbar = tqdm(range(int(num_epochs)) , unit="epoch", mininterval=0, disable=False, desc='train')
    num_batches = x_train.shape[0] // batch_size #throw away last incomplete batch

    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    best_validation_loss = torch.inf
    best_validation_acc = torch.inf
    best_net = copy.deepcopy(net)

    for i in pbar:
        net.train()
        #random permutation of data
        # rand_idxs = torch.randperm(x_train.shape[0])

        # rand_idxs_aux = torch.randperm(x_train_aux.shape[0])
        avg_loss = 0.0
        avg_acc = 0.0
        
        # for batch_num in range(0, num_batches):
            # batch_idxs = rand_idxs[batch_num*batch_size: (batch_num+1)*batch_size]
            
            # x_batch = x_train[batch_idxs].to(device)
            # y_batch = y_train[batch_idxs].to(device)

        for data in trainloader:
            x_batch = data[0]
            y_batch = data[1]
            if auxtrainloader is not None:
                aux_data = next(iter(auxtrainloader))
                x_batch_aux = aux_data[0]
                y_batch_aux = aux_data[1]
                x_batch = torch.cat((x_batch, x_batch_aux), dim=0)
                y_batch = torch.cat((y_batch, y_batch_aux), dim=0)

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)            
            y_pred = net.forward(x_batch)
            loss = loss_fn(y_pred, y_batch)

            # batch_idxs_aux = rand_idxs_aux[batch_num*batch_size: (batch_num+1)*batch_size]
            # x_batch_aux = x_train_aux[batch_idxs_aux].to(device)
            # y_batch_aux = y_train_aux[batch_idxs_aux].to(device)

            # y_pred_aux = net.forward(x_batch_aux)
            # loss_aux = loss_fn(y_pred_aux, y_batch_aux)
            # loss += loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            if is_classifier:
                acc = (torch.sigmoid(y_pred).round() == y_batch).float().mean().cpu()
                avg_acc += acc.item()
        
        avg_loss /= num_batches*1.0
        avg_acc /= num_batches*1.0
        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)
        
        #run validation
        net.eval()
        with torch.no_grad():
            y_pred_val = net.forward(x_val)
            loss_val = loss_fn(y_pred_val, y_val)
            validation_losses.append(loss_val.item())
            acc_val = torch.tensor([0.0])
            if is_classifier:
                acc_val = (torch.sigmoid(y_pred_val).round() == y_val).float().mean().cpu()
                validation_accuracies.append(acc_val.item())
        
            if loss_val.item() <= best_validation_loss:
                print('Best model so far. Val Loss: {}'.format(loss_val.item()))
                best_validation_loss = loss_val
                best_net = copy.deepcopy(net)
        

        pbar.set_postfix(
            avg_loss_train=float(avg_loss),
            avg_acc_train=float(avg_acc),
            avg_loss_val = float(loss_val.item()),
            avg_acc_val = float(acc_val.item())
        )

    return best_net, train_losses, train_accuracies, validation_losses, validation_accuracies, norm_dict
    
def scale_to_net(data, norm_dict, key):
    """Scale the tensor network range

    Args:
        data (tensor): input tensor to scale
        norm_dict (Dict): normalization dictionary of the form dict={key:{'mean':,'std':}}
        key (str): key of the data

    Returns:
        tensor : output scaled tensor
    """    
    
    scaled_data = torch.div(data - norm_dict[key]['mean'], norm_dict[key]['std'])
    scaled_data[scaled_data != scaled_data] = 0.0
    return scaled_data


def scale_to_base(data, norm_dict, key):
    """Scale the tensor back to the orginal units.  

    Args:
        data (tensor): input tensor to scale
        norm_dict (Dict): normalization dictionary of the form dict={key:{'mean':,'std':}}
        key (str): key of the data

    Returns:
        tensor : output scaled tensor
    """    
    scaled_data = torch.mul(data, norm_dict[key]['std']) + norm_dict[key]['mean']
    return scaled_data


def cat_dict_list(input_dict_list, idx):
    #TODO: Handle the case where no idxs are provided
    D = {}
    d_0 = input_dict_list[0]
    K = list(d_0.keys())

    for k in K:
        if isinstance(d_0[k], dict):
            l = [d[k] for d in input_dict_list]
            D[k] = cat_dict_list(l, idx)
        else:
            if d_0[k].ndim > 1:
                D[k] = torch.cat([d[k][idx].unsqueeze(0) for d in input_dict_list], dim=0)
            else:
                D[k] = torch.tensor([d[k][idx] for d in input_dict_list], device=d_0[k].device, dtype=d_0[k].dtype)
    
    return D


def run_episode(
        env, task, policy, 
        max_episode_steps:int,
        deterministic: bool = False,
        compute_cost: bool = True,
        compute_termination: bool = True,
        discount: float = 0.99,
        rng:Optional[torch.Generator] = None):
        
        reset_data, state = None, None
        try: 
            obs, reset_info = env.reset(rng=rng)
            state = reset_info['state']
            reset_data = reset_info['reset_data']
            done = reset_info['done']
            policy.reset(reset_data)
        except:
            obs = env.reset()
            done = False

        traj_data = defaultdict(list)
        total_return, discounted_total_return = 0.0, 0.0

        for i in range(max_episode_steps):
            with torch.no_grad():
                policy_input = {
                    'obs': torch.as_tensor(obs).float(),
                    'states': state}
                
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                action, policy_info = policy.get_action(policy_input, deterministic=deterministic)

                # if i == 4:
                #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
                #     exit()

                #step tells me about next state
                next_obs, next_reward, next_done, info = env.step(
                    action, compute_cost=compute_cost, compute_termination=compute_termination)

                next_state = info['state'] if 'state' in info else None
                total_return += next_reward
                discounted_total_return += next_reward * discount**i

            timeout = i == max_episode_steps - 1
            terminal = done and (not timeout)

            traj_data['observations'].append(obs)
            traj_data['actions'].append(action)
            traj_data['next_observations'].append(next_obs)
            traj_data['rewards'].append(next_reward)
            traj_data['terminals'].append(bool(terminal))
            traj_data['timeouts'].append(bool(timeout))
            if reset_data is not None:
                goal_dict = reset_data['goal_dict']
                for k,v in goal_dict.items(): traj_data['goals/'+k].append(v)
            if state is not None:
                for k,v in state.items():traj_data['states/'+k].append(v.squeeze(-1))
            if next_state is not None:
                for k,v in next_state.items():traj_data['next_states/'+k].append(v.squeeze(-1))
        
            # traj_data['qpos'].append(qpos)
            # traj_data['qvel'].append(qvel)

            if done or timeout: break
            obs = copy.deepcopy(next_obs)
            state = copy.deepcopy(next_state)
            done = next_done

        for k in traj_data:
            if torch.is_tensor(traj_data[k][0]):
                traj_data[k] = torch.cat(traj_data[k], dim=0).cpu().numpy()
            else: traj_data[k] = np.array(traj_data[k])

        traj_info = {
            'return': total_return,
            'discounted_return': discounted_total_return,
            'traj_length': i+1
            }

        
        return traj_data, traj_info

def evaluate_policy(
        env, task, policy, 
        max_episode_steps:int,
        num_episodes:int,
        deterministic: bool = False,
        compute_cost: bool = True,
        compute_termination: bool = True,
        discount: float = 0.99,
        normalize_score_fn = None,                                     
        rng:Optional[torch.Generator] = None):

    ep_datas = []
    ep_infos = []
    num_steps = 0
    for ep_num in range(num_episodes):
        ep_data, ep_info = run_episode(
        env, task, policy, 
        max_episode_steps,
        deterministic,
        compute_cost,
        compute_termination,
        discount, rng)
        ep_datas.append(ep_data)
        ep_infos.append(ep_info)
        num_steps += ep_info['traj_length']
   
    eval_returns = np.array([ep_info['return'] for ep_info in ep_infos])
    eval_discounted_returns = np.array([ep_info['discounted_return'] for ep_info in ep_infos])

    info = {
        'Eval/mean_return': eval_returns.mean(),
        'Eval/std_return': eval_returns.std(),
        'Eval/discounted_return': eval_discounted_returns.mean(),
        'Eval/num_steps': num_steps}

    if normalize_score_fn is not None:
        normalized_returns = normalize_score_fn(eval_returns)
        info['Eval/normalized_return_mean'] = normalized_returns.mean()
        info['Eval/normalized_return_std'] = normalized_returns.std()

    return ep_datas, info


def preprocess_dataset(train_dataset, env, task=None, cfg=None, normalize_score_fn=None):
    info = {}
    # discount = 1.0
    validation_dataset = None
    # if cfg is not None:
    discount = cfg.discount
    #relabel cost, rewards and terminals
    if task is not None and cfg.relabel_data:
        train_dataset = relabel_dataset(train_dataset, task)
    #add to dataset returns and quantities required for MC targets
    train_dataset, ep_returns, ep_disc_returns = add_returns_to_dataset(train_dataset, discount=discount)
    #split into training and validation trajectories
    if cfg.train_val_split_ratio > 0.:
        train_dataset, validation_dataset = train_val_split(train_dataset, cfg.train_val_split_ratio)

    #Set range of rewards/costs
    rew_or_cost = train_dataset['costs'] if 'costs' in train_dataset else train_dataset['rewards']
    info['r_c_min'] = rew_or_cost.min().item()
    info['r_c_max'] = rew_or_cost.max().item()
    info['r_c_mean'] = rew_or_cost.mean().item()
    info['r_c_std'] = rew_or_cost.std().item()
    # Set range of value functions
    # TODO: Check if this is right since we should take episode length into account too
    # if cfg.clip_values:
    Vmax = max(0.0, rew_or_cost.max()/(1.-discount)).item()
    Vmin = min(0.0, rew_or_cost.min()/(1.-discount), Vmax-1.0/(1-discount))
    info['V_max'] = Vmax
    info['V_min'] = Vmin
    
    #extract dataset of "success states" from train_dataset
    #TODO: Maybe this should just be a dataset of trajectories with 
    # terminals in them 
    terminal_dataset=None
    if "terminals" in train_dataset.keys():
        terminal_idxs = train_dataset["terminals"] > 0
        terminal_dataset = ReplayBuffer(capacity=len(train_dataset["terminals"].nonzero()), device=train_dataset.device)
        terminal_batch = {k: v[terminal_idxs] for (k,v) in train_dataset.items()}
        terminal_dataset.add_batch(terminal_batch)

    #Get ranges of returns/disc returns
    info["return_min"] = min(train_dataset["returns"]).item()
    info["return_max"] = max(train_dataset["returns"]).item()
    info["return_mean"] = train_dataset["returns"].mean().item()
    info["return_std"] = train_dataset["returns"].std().item()
    info["disc_return_max"] = max(train_dataset["disc_returns"]).item()
    info["disc_return_min"] = min(train_dataset["disc_returns"]).item()
    info["disc_return_mean"] = train_dataset["disc_returns"].mean().item()
    info["disc_return_std"] = train_dataset["disc_returns"].std().item() + 1e-12
    info['ep_return_min'] = min(ep_returns)
    info['ep_return_max'] = max(ep_returns)
    info['ep_disc_return_min'] = min(ep_disc_returns)
    info['ep_disc_return_max'] = max(ep_disc_returns)
    
    #Get stats of observations (for normalization)
    info['obs_mean'] = train_dataset["observations"].mean(0)
    info['obs_std'] = train_dataset["observations"].std(0) + 1e-12
    info['obs_max'] = train_dataset["observations"].max(0)[0]
    info['obs_min'] = train_dataset["observations"].min(0)[0]

    if normalize_score_fn is not None:
        normalized_returns = normalize_score_fn(np.array(train_dataset['returns']))
        info['normalized_return_mean'] = normalized_returns.mean()
        info['normalized_return_std'] = normalized_returns.std()
    
    return train_dataset, validation_dataset, terminal_dataset, info



def add_returns_to_dataset(dataset, discount):

    #Augment data with returns/discounted returns and other quantitites 
    # useful for calculating MC targets
    returns, ep_returns = [], []
    disc_returns, ep_disc_returns = [], []
    remaining_steps, last_observations = [], []
    last_terminals = []
    for ep in dataset.episode_iterator():
        episode_r_c = ep['costs'] if 'costs' in ep else ep['rewards']
        H = len(episode_r_c)
        ret = discount_cumsum(episode_r_c, 1.0)
        disc_ret = discount_cumsum(episode_r_c, discount)
        returns.append(ret)
        disc_returns.append(disc_ret)
        ep_returns.append(ret[0].item())
        ep_disc_returns.append(disc_ret[0].item())
        #Note this only works for torch datasets
        rem_steps = torch.flip(torch.arange(H, device=dataset.device), dims=(0,))+1
        assert rem_steps.shape == episode_r_c.shape
        remaining_steps.append(rem_steps)
        last_obs = torch.repeat_interleave(ep['observations'][-1:], H, dim=0)
        assert last_obs.shape == ep['observations'].shape
        last_observations.append(last_obs) 
        last_term = torch.repeat_interleave(ep['terminals'][-1], H)
        assert last_term.shape == ep['terminals'].shape
        last_terminals.append(last_term)

    returns = torch.cat(returns, dim=0)
    disc_returns = torch.cat(disc_returns, dim=0)
    dataset['returns'] = returns
    dataset['disc_returns'] = disc_returns
    dataset['remaining_steps'] = torch.cat(remaining_steps, dim=0)
    dataset['last_observations'] = torch.cat(last_observations, dim=0)
    dataset['last_terminals'] = torch.cat(last_terminals, dim=0)

    return dataset, ep_returns, ep_disc_returns


def relabel_dataset(dataset, task):
    device = task.device
    q_pos = torch.as_tensor(dataset['states/q_pos']).to(device)
    q_vel = torch.as_tensor(dataset['states/q_vel']).to(device)
    q_acc = torch.as_tensor(dataset['states/q_acc']).to(device)
    actions = torch.as_tensor(dataset['actions']).to(device)
    
    state_dict = {'q_pos': q_pos, 'q_vel': q_vel, 'q_acc': q_acc}
    goal_dict = {}
    for k,v in dataset.items():
        split = k.split("/")
        if split[0] == 'goals':
            goal_dict[split[-1]] = torch.as_tensor(v).to(device)
    
    with torch.no_grad():
        task.update_params({'goal_dict': goal_dict})
        full_state_dict = task.compute_full_state(state_dict)
        new_cost, cost_terms = task.compute_cost(full_state_dict, actions)
        new_terminals, new_terminal_cost, term_info = task.compute_termination(full_state_dict)
        new_cost += new_terminal_cost
        new_observations = task.compute_observations(full_state_dict, compute_full_state=False, cost_terms=cost_terms)
        new_success = task.compute_success(full_state_dict)

    dataset["observations"] = new_observations.to(dataset.device) #TODO Neel - Relabel terminal state observations
    dataset["costs"] = new_cost.to(dataset.device)
    dataset["terminals"] = new_terminals.to(dataset.device)
    dataset["success"] = new_success.to(dataset.device)    
    return dataset

def plot_episode(episode, block=False):

    q_pos = episode['states/q_pos']
    q_vel = episode['states/q_vel']
    q_acc = episode['states/q_acc']
    actions = episode['actions']
    costs = episode['costs']
    returns = episode['returns'] if 'returns' in episode else None
    disc_returns = episode['disc_returns'] if 'disc_returns' in episode else None
    if torch.is_tensor(q_pos): 
        q_pos = q_pos.cpu().numpy()
    if torch.is_tensor(q_vel): 
        q_vel = q_vel.cpu().numpy()
    if torch.is_tensor(q_acc): 
        q_acc = q_acc.cpu().numpy()
    if torch.is_tensor(actions): 
        actions = actions.cpu().numpy()
    if torch.is_tensor(costs):
        costs = costs.cpu().numpy()
    if returns is not None and torch.is_tensor(returns):
        returns = returns.cpu().numpy()
    if disc_returns is not None and torch.is_tensor(disc_returns):
        disc_returns = disc_returns.cpu().numpy()
    if ee_goal is not None and torch.is_tensor(ee_goal):
        ee_goal = ee_goal.cpu().numpy()


    fig, ax = plt.subplots(3,1)
    num_points, n_dofs = q_pos.shape
    for n in range(n_dofs):
        ax[0].plot(q_pos[:,n], label='dof_{}'.format(n+1))
        ax[1].plot(q_vel[:,n])
        ax[2].plot(actions[:,n])
    
    ax[-1].set_xlabel('Episode Timestep')
    ax[0].set_ylabel('q_pos (rad)')
    ax[1].set_ylabel('q_vel (rad/s)')
    ax[2].set_ylabel('actions [q_acc] (rad/s^2)')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True, shadow=True)    

    fig2, ax2 = plt.subplots(3,1)
    ax2[0].plot(costs)
    if returns is not None: ax2[1].plot(returns)
    if disc_returns is not None: ax2[2].plot(disc_returns)

    ax2[0].set_ylabel('Costs')
    ax2[1].set_ylabel('Returns')
    ax2[2].set_ylabel('Discounted Returns')
    ax2[-1].set_xlabel('Episode Timestep')

    plt.show(block=block)
    if not block:
        plt.waitforbuttonpress(-1)
        plt.close(fig)
        plt.close(fig2)

# Below are modified from 
# https://github.com/gwthomas/IQL-PyTorch

def discount_cumsum(x, discount):
    """Discounted cumulative sum.
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    Here, we have y[t] - discount*y[t+1] = x[t]
    or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    Args:
        x (np.ndarrary): Input.
        discount (float): Discount factor.
    Returns:
        np.ndarrary: Discounted cumulative sum.
    """
    if torch.is_tensor(x):
        return torchaudio.functional.lfilter(
                x.flip(dims=(0,)),
                a_coeffs=torch.tensor([1, -discount], device=x.device),
                b_coeffs=torch.tensor([1, 0], device=x.device), clamp=False).flip(dims=(0,))
    else:
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=-1)[::-1]

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias




#DEPRECATED
    

# def episode_runner(
#         envs,
#         num_episodes: int, 
#         policy,
#         task,
#         collect_data: bool = False,
#         deterministic: bool = False,
#         debug: bool = False,
#         compute_termination: bool = True,
#         device: torch.device = torch.device('cpu'),
#         rng:Optional[torch.Generator] = None):  
        
#         buffer = None
#         if collect_data:
#             buffer = ReplayBuffer(capacity=int(1e6), device=device)
        
#         total_steps_collected = 0

#         reset_data = task.reset(rng=rng)
#         policy.reset(reset_data)
#         curr_state_dict = copy.deepcopy(envs.reset(reset_data))
#         # curr_obs = task.forward(curr_state_dict)[0]
#         # obs, state_dict_full = task.compute_observations(state_dict=state_dict)
#         # curr_obs = curr_obs.view(envs.num_envs, obs_dim)
#         # curr_costs = torch.zeros(envs.num_envs, device=device)
#         # episode_lens = torch.zeros(envs.num_envs, device=device)
#         # avg_episode_cost = 0.0
#         # episodes_terminated = 0
#         # episodes_done = 0
#         num_episodes_done = 0
#         num_episodes_terminated = 0
#         episode_lens = 0
#         # transition_dict_list = []
#         # episode_metrics_list = []
#         # episode_cost_buffer = []
#         print('Collecting {0} episodes, determinsitic={1}'.format(num_episodes, deterministic))
#         while num_episodes_done < num_episodes:
#             with torch.no_grad():

#                 policy_input = {
#                     'states': curr_state_dict}
                                
#                 command, policy_info = policy.get_action(policy_input, deterministic=deterministic)
#                 actions = policy_info['action']
#                 curr_filtered_state = policy_info['filtered_states']
#                 if actions.ndim == 3:
#                     actions = actions.squeeze(0)

#                 next_state_dict, done_env = envs.step(command)
#                 # next_obs, cost, done_task, cost_terms, done_cost, done_info = task.forward(next_state_dict, actions)
#                 done_task = torch.zeros_like(done_env)
#                 if compute_termination:
#                     done_task, done_cost, done_info = task.compute_termination(
#                         curr_state_dict, actions, compute_full_state=True)
#                     if done_task.item() > 0:
#                         print(done_info['in_bounds'])
#                         print('state_dict', curr_state_dict)
#                         print('state_bounds', task.state_bounds)
#                         print('command', command)
#                 # done_task = done_task.view(envs.num_envs,)
#                 # cost = cost.view(envs.num_envs,)
#                 # done_cost = done_cost.view(envs.num_envs,)
#                 # task.update_state(next_state_dict)
#                 # next_obs = task.compute_observations()
#                 # next_obs = next_obs.view(envs.num_envs, obs_dim)

#                 # next_obs, next_state_dict_full = task.compute_observations(next_state_dict)
#                 # done_task, done_cost, _ = task.compute_termination(state_dict_full, actions)
#                 # cost, _, _ = task.compute_cost(state_dict=state_dict_full, action_batch=actions, termination_cost=done_cost)
#                 # obs, cost, done_task, cost_terms, done_cost = task.forward(next_state_dict, actions)


#                 # next_obs = next_obs.view(envs.num_envs, obs_dim)
#                 # done_task = done_task.view(envs.num_envs,)
#                 # cost = cost.view(envs.num_envs,)
#                 # done_cost = done_cost.view(envs.num_envs, )
                
#             # curr_costs += cost
#             done = (done_env + done_task) > 0

#             #remove timeout from done
#             episode_lens += 1
#             timeout = episode_lens == envs.max_episode_length - 1
#             done_without_timeouts = done * (1.0-timeout)

#             total_steps_collected += 1
#             episode_done = done.item()
#             num_episodes_terminated += done_without_timeouts.item() 

#             transition_dict = {}  
#             for k in curr_state_dict:
#                 transition_dict['states/{}'.format(k)] = copy.deepcopy(curr_state_dict[k])              
#             for k in curr_filtered_state:
#                 transition_dict['states/filtered/{}'.format(k)] = copy.deepcopy(curr_filtered_state[k])              
#             for k in reset_data['goal_dict']:
#                 transition_dict['goal/{}'.format(k)] = reset_data['goal_dict'][k]
#             # transition_dict['state_dict'] = copy.deepcopy(curr_state_dict)
#             # transition_dict['next_state_dict'] = copy.deepcopy(next_state_dict)
#             # transition_dict['filtered_state_dict'] = copy.deepcopy(curr_filtered_state)
#             # transition_dict['goal_dict'] = reset_data['goal_dict']
#             transition_dict['actions'] = copy.deepcopy(actions)
#             # transition_dict['obs'] = curr_obs.clone()
#             # transition_dict['next_obs'] = next_obs.clone()
#             # transition_dict['cost'] = cost
#             transition_dict['terminals'] = done_without_timeouts
#             transition_dict['timeouts'] = torch.tensor([timeout])     
#             # transition_dict_list.append(transition_dict)  
#             if collect_data:
#                 buffer.add(transition_dict)

#             curr_state_dict = copy.deepcopy(next_state_dict)
#             # curr_obs = next_obs.clone()
#             # state_dict_full = copy.deepcopy(next_state_dict_full)

#             #reset if done
#             # done_indices = done.nonzero(as_tuple=False).squeeze(-1)
#             # done_episode_costs = curr_costs[done_indices]
#             # curr_num_eps_done = len(done_indices)

#             # curr_num_eps_terminated = torch.sum(done_without_timeouts).item()

#             # episodes_done += curr_num_eps_done
#             # episodes_terminated += curr_num_eps_terminated
#             # curr_num_steps = cost.shape[0]

#             # if curr_num_eps_done > 0:
#             if episode_done:
#                 num_episodes_done += 1
#                 episode_lens = 0
#                 #Add done episode to buffer
#                 # for idx in done_indices:
#                     # episode_dict = cat_dict_list(transition_dict_list, idx)

#                     # if update_buffer:
#                     #     buffer.add(episode_dict)
                    
#                     #compute episode metrics
#                     # episode_metrics_list.append(task.compute_metrics(episode_dict))
#                     # episode_cost_buffer.append(curr_costs[idx].item())
                                    
#                 #Reset everything
#                 # reset_data = task.reset_idx(done_indices, rng=rng)
#                 # curr_state_dict = envs.reset_idx(done_indices, reset_data)
#                 reset_data = task.reset(rng=rng)
#                 curr_state_dict = envs.reset(reset_data)
#                 #TODO: policy should be reset only for the required instances
#                 #especially this is true for MPC policies
#                 policy.reset(reset_data)
#                 # task.update_state(curr_state_dict)
#                 # obs=task.compute_observations()
#                 # curr_obs = task.forward(curr_state_dict)[0]
#                 # obs, state_dict_full = task.compute_observations(state_dict=state_dict)
#                 # curr_obs = curr_obs.view(envs.num_envs, obs_dim)
#                 # transition_dict_list = [] #TODO: This also needs to be reset for specific instances
                
#             # curr_num_eps_dones = torch.sum(done).item()
#             # if curr_num_eps_dones > 0:
#                 # for i in range(curr_num_eps_dones):
#                 #     episode_cost_buffer.append(done_episode_costs[i].item())
#                     # if len(episode_reward_buffer) > 10:
#                     #     episode_reward_buffer.pop(0)

#             #Reset costs and episode_lens for episodes that are done only
#             # not_done = 1.0 - done.float()
#             # curr_costs = curr_costs * not_done
#             # episode_lens = episode_lens * not_done

#         # if len(episode_cost_buffer) > 0:
#         #     avg_episode_cost = np.average(episode_cost_buffer).item()

#         metrics = {
#             'num_steps_collected': total_steps_collected,
#             'num_eps_collected': num_episodes_done,
#             'num_eps_terminated': num_episodes_terminated,
#             }
        
#         return buffer, metrics


# if isinstance(episode_r_c, torch.Tensor):
#     ret = torch.cumsum(episode_r_c.flip(0), dim=0).flip(0)
# else:
#     ret = np.cumsum(episode_r_c.flip(0)).flip(0)

# def minimal_episode_runner(
#     envs,
#     num_episodes: int, 
#     policy,
#     task,
#     buffer: Optional[ReplayBuffer] = None,
#     deterministic: bool = False,
#     debug: bool = False,
#     device: torch.device = torch.device('cpu'),
#     rng: Optional[torch.Generator] = None):        
    
#     update_buffer = False
#     if buffer is not None:
#         update_buffer = True
#     # obs_dim = task.obs_dim
#     total_steps_collected = 0

#     reset_data = task.reset(rng=rng)
#     policy.reset(reset_data)
#     curr_state_dict = copy.deepcopy(envs.reset(reset_data))
#     # curr_obs = task.forward(curr_state_dict)[0]
#     # obs, state_dict_full = task.compute_observations(state_dict=state_dict)
#     # curr_obs = curr_obs.view(envs.num_envs, obs_dim)
#     # curr_costs = torch.zeros(envs.num_envs, device=device)
#     episode_lens = torch.zeros(envs.num_envs, device=device)
    
#     # avg_episode_cost = 0.0
#     episodes_terminated = 0
#     episodes_done = 0
#     transition_dict_list = []
#     episode_metrics_list = []
#     episode_cost_buffer = []

#     while episodes_done < num_episodes:

#         with torch.no_grad():

#             policy_input = {
#                 'states': curr_state_dict}
                            
#             command, policy_info = policy.get_action(policy_input, deterministic=deterministic)


#             actions = policy_info['action']
#             curr_filtered_state = policy_info['filtered_states']

#             if actions.ndim == 3:
#                 actions = actions.squeeze(0)

#             next_state_dict, done_env = envs.step(command)
            
#             # next_obs, cost, done_task, cost_terms, done_cost, term_info = task.forward(next_state_dict, actions)
#             # done_task = done_task.view(envs.num_envs,)
#             # cost = cost.view(envs.num_envs,)
#             # done_cost = done_cost.view(envs.num_envs,)
#             # next_obs = next_obs.view(envs.num_envs, obs_dim)

#             # if debug:
#             #     pass
       
#         # curr_costs += cost
#         episode_lens += 1
#         done = (done_env) > 0 #+ done_task

#         #remove timeout from done
#         timeout = episode_lens == envs.max_episode_length - 1
#         done_without_timeouts = done * (1.0-timeout)

#         transition_dict = {}
#         transition_dict['state_dict'] = copy.deepcopy(curr_state_dict)
#         transition_dict['next_state_dict'] = copy.deepcopy(next_state_dict)
#         transition_dict['filtered_state_dict'] = copy.deepcopy(curr_filtered_state)

#         transition_dict['goal_dict'] = reset_data['goal_dict']
#         transition_dict['actions'] = copy.deepcopy(actions)
#         # transition_dict['obs'] = curr_obs.clone()
#         # transition_dict['next_obs'] = next_obs.clone()
#         # transition_dict['cost'] = cost
#         transition_dict['done'] = done_without_timeouts
#         transition_dict['timeout'] = timeout
        
#         transition_dict_list.append(transition_dict)
        
#         curr_state_dict = copy.deepcopy(next_state_dict)
#         # curr_obs = next_obs.clone()

#         #reset if done
#         done_indices = done.nonzero(as_tuple=False).squeeze(-1)
#         # done_episode_costs = curr_costs[done_indices]
#         curr_num_eps_done = len(done_indices)
#         curr_num_eps_terminated = torch.sum(done_without_timeouts).item()

#         episodes_done += curr_num_eps_done
#         episodes_terminated += curr_num_eps_terminated

#         # curr_num_steps = cost.shape[0]
#         total_steps_collected += envs.num_envs

#         if curr_num_eps_done > 0:
#             #Add done episode to buffer
#             episode_dict = cat_dict_list(transition_dict_list, 0)

#             if update_buffer:
#                 buffer.add(episode_dict)
                
#             #compute episode metrics
#             # episode_metrics_list.append(task.compute_metrics(episode_dict))
#             # episode_cost_buffer.append(curr_costs[0].item())
                                
#             #Reset everything
#             reset_data = task.reset_idx(done_indices, rng=rng)
#             curr_state_dict = envs.reset(reset_data)
#             policy.reset(reset_data)
#             # curr_obs = task.forward(curr_state_dict)[0]
#             # curr_obs = curr_obs.view(envs.num_envs, obs_dim)
#             transition_dict_list = []
            
#         #Reset costs and episode_lens for episodes that are done only
#         not_done = 1.0 - done.float()
#         # curr_costs = curr_costs * not_done
#         episode_lens = episode_lens * not_done

#     # if len(episode_cost_buffer) > 0:
#     #     avg_episode_cost = np.average(episode_cost_buffer).item()

#     #Consolidate emtrics to be returned        
#     metrics = {
#         'num_steps_collected': total_steps_collected,
#         'num_eps_completed': episodes_done,
#         'num_eps_terminated': episodes_terminated,
#         # 'avg_episode_cost': avg_episode_cost,
#         }
    
#     if buffer is not None:
#         metrics['buffer_size'] = len(buffer)

#     # episode_metrics_keys = episode_metrics_list[0].keys()
#     # for k in episode_metrics_keys:
#     #     avg_val = np.average([m[k] for m in episode_metrics_list]).item()
#     #     metrics[k] = avg_val

#     return buffer, metrics
