import csv, json, os, random, string, sys
import numpy as np
import copy
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from typing import Optional
import time
from torch.utils.data import TensorDataset, DataLoader
from storm_kit.learning.replay_buffers import RobotBuffer


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    # rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}' #_{rand_str}'

# def concatenate_dicts(dict1, dict2):
#     dict_cat = {}
#     for k, v in dict1.items():
#         if isinstance(v, dict):

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
    buffer = RobotBuffer(capacity=1000)
    buffer.load(filepath)
    print('Loaded buffer {}'.format(buffer))
    return buffer, len(buffer)

def buffer_from_folder(data_dir, capacity=None, device=torch.device('cpu')):
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
    buffer = RobotBuffer(buffer_capacity, 7, device)
    for b in episode_buffers:
        buffer.concatenate(b.state_dict()) 
    return buffer


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

def episode_runner(
        envs,
        num_episodes: int, 
        policy,
        task,
        buffer: Optional[RobotBuffer] = None,
        deterministic: bool = False,
        debug: bool = False,
        device: torch.device = torch.device('cpu')):        
        
        update_buffer = False
        if buffer is not None:
            update_buffer = True
        
        obs_dim = task.obs_dim
        total_steps_collected = 0

        reset_data = task.reset()
        policy.reset(reset_data)
        curr_state_dict = copy.deepcopy(envs.reset(reset_data))
        curr_obs = task.forward(curr_state_dict)[0]
        # obs, state_dict_full = task.compute_observations(state_dict=state_dict)
        curr_obs = curr_obs.view(envs.num_envs, obs_dim)
        curr_costs = torch.zeros(envs.num_envs, device=device)
        episode_lens = torch.zeros(envs.num_envs, device=device)
        
        avg_episode_cost = 0.0
        episodes_terminated = 0
        episodes_done = 0
        transition_dict_list = []
        episode_metrics_list = []
        episode_cost_buffer = []

        while episodes_done < num_episodes:

            with torch.no_grad():

                policy_input = {
                    'states': curr_state_dict,
                    'obs': curr_obs}
                                
                command, policy_info = policy.get_action(policy_input, deterministic=deterministic)


                actions = policy_info['action']
                if actions.ndim == 3:
                    actions = actions.squeeze(0)

                next_state_dict, done_env = envs.step(command)
                
                next_obs, cost, done_task, cost_terms, done_cost, done_info = task.forward(next_state_dict, actions)
                print(done_info)
                done_task = done_task.view(envs.num_envs,)
                cost = cost.view(envs.num_envs,)
                done_cost = done_cost.view(envs.num_envs,)
                # task.update_state(next_state_dict)
                # next_obs = task.compute_observations()
                next_obs = next_obs.view(envs.num_envs, obs_dim)

                if debug:
                    pass

                # next_obs, next_state_dict_full = task.compute_observations(next_state_dict)
                # done_task, done_cost, _ = task.compute_termination(state_dict_full, actions)
                # cost, _, _ = task.compute_cost(state_dict=state_dict_full, action_batch=actions, termination_cost=done_cost)
                # obs, cost, done_task, cost_terms, done_cost = task.forward(next_state_dict, actions)


                # next_obs = next_obs.view(envs.num_envs, obs_dim)
                # done_task = done_task.view(envs.num_envs,)
                # cost = cost.view(envs.num_envs,)
                # done_cost = done_cost.view(envs.num_envs, )
                
            curr_costs += cost
            episode_lens += 1
            done = (done_env + done_task) > 0

            #remove timeout from done
            timeout = episode_lens == envs.max_episode_length - 1
            done_without_timeouts = done * (1-timeout.float())

            transition_dict = {}
            transition_dict['state_dict'] = copy.deepcopy(curr_state_dict)
            transition_dict['next_state_dict'] = copy.deepcopy(next_state_dict)
            transition_dict['goal_dict'] = reset_data['goal_dict']
            transition_dict['actions'] = copy.deepcopy(actions)
            transition_dict['obs'] = curr_obs.clone()
            transition_dict['next_obs'] = next_obs.clone()
            transition_dict['cost'] = cost
            transition_dict['done'] = done_without_timeouts
            transition_dict['timeout'] = timeout
            
            transition_dict_list.append(transition_dict)
            
            # if update_buffer:
            #     buffer.add(transition_dict)

            curr_state_dict = copy.deepcopy(next_state_dict)
            curr_obs = next_obs.clone()
            # state_dict_full = copy.deepcopy(next_state_dict_full)

            #reset if done
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)
            done_episode_costs = curr_costs[done_indices]
            curr_num_eps_done = len(done_indices)
            curr_num_eps_terminated = torch.sum(done_without_timeouts).item()

            episodes_done += curr_num_eps_done
            episodes_terminated += curr_num_eps_terminated
            curr_num_steps = cost.shape[0]
            total_steps_collected += curr_num_steps

            if curr_num_eps_done > 0:
                #Add done episode to buffer
                for idx in done_indices:
                    episode_dict = cat_dict_list(transition_dict_list, idx)

                    if update_buffer:
                        buffer.add(episode_dict)
                    
                    #compute episode metrics
                    episode_metrics_list.append(task.compute_metrics(episode_dict))
                    episode_cost_buffer.append(curr_costs[idx].item())
                                    
                #Reset everything
                reset_data = task.reset_idx(done_indices)
                curr_state_dict = envs.reset_idx(done_indices, reset_data)
                #TODO: policy should be reset only for the required instances
                #especially this is true for MPC policies
                policy.reset(reset_data)
                # task.update_state(curr_state_dict)
                # obs=task.compute_observations()
                curr_obs = task.forward(curr_state_dict)[0]
                # obs, state_dict_full = task.compute_observations(state_dict=state_dict)
                curr_obs = curr_obs.view(envs.num_envs, obs_dim)
                transition_dict_list = [] #TODO: This also needs to be reset for specific instances
                
            # curr_num_eps_dones = torch.sum(done).item()
            # if curr_num_eps_dones > 0:
                # for i in range(curr_num_eps_dones):
                #     episode_cost_buffer.append(done_episode_costs[i].item())
                    # if len(episode_reward_buffer) > 10:
                    #     episode_reward_buffer.pop(0)

            #Reset costs and episode_lens for episodes that are done only
            not_done = 1.0 - done.float()
            curr_costs = curr_costs * not_done
            episode_lens = episode_lens * not_done

        if len(episode_cost_buffer) > 0:
            avg_episode_cost = np.average(episode_cost_buffer).item()

        #Consolidate emtrics to be returned        
        metrics = {
            'num_steps_collected': total_steps_collected,
            'num_eps_completed': episodes_done,
            'num_eps_terminated': episodes_terminated,
            'avg_episode_cost': avg_episode_cost,
            }
        
        if buffer is not None:
            metrics['buffer_size'] = len(buffer)

        episode_metrics_keys = episode_metrics_list[0].keys()
        for k in episode_metrics_keys:
            avg_val = np.average([m[k] for m in episode_metrics_list]).item()
            metrics[k] = avg_val

        return buffer, metrics


def minimal_episode_runner(
    envs,
    num_episodes: int, 
    policy,
    task,
    buffer: Optional[RobotBuffer] = None,
    deterministic: bool = False,
    debug: bool = False,
    device: torch.device = torch.device('cpu')):        
    
    update_buffer = False
    if buffer is not None:
        update_buffer = True
    
    obs_dim = task.obs_dim
    total_steps_collected = 0

    reset_data = task.reset()
    policy.reset(reset_data)
    curr_state_dict = copy.deepcopy(envs.reset(reset_data))
    curr_obs = task.forward(curr_state_dict)[0]
    # obs, state_dict_full = task.compute_observations(state_dict=state_dict)
    curr_obs = curr_obs.view(envs.num_envs, obs_dim)
    curr_costs = torch.zeros(envs.num_envs, device=device)
    episode_lens = torch.zeros(envs.num_envs, device=device)
    
    avg_episode_cost = 0.0
    episodes_terminated = 0
    episodes_done = 0
    transition_dict_list = []
    episode_metrics_list = []
    episode_cost_buffer = []

    while episodes_done < num_episodes:

        with torch.no_grad():

            policy_input = {
                'states': curr_state_dict,
                'obs': curr_obs}
                            
            command, policy_info = policy.get_action(policy_input, deterministic=deterministic)


            actions = policy_info['action']
            if actions.ndim == 3:
                actions = actions.squeeze(0)

            next_state_dict, done_env = envs.step(command)
            
            next_obs, cost, done_task, cost_terms, done_cost, term_info = task.forward(next_state_dict, actions)
            done_task = done_task.view(envs.num_envs,)
            cost = cost.view(envs.num_envs,)
            done_cost = done_cost.view(envs.num_envs,)
            next_obs = next_obs.view(envs.num_envs, obs_dim)

            if debug:
                pass
       
        curr_costs += cost
        episode_lens += 1
        done = (done_env + done_task) > 0

        #remove timeout from done
        timeout = episode_lens == envs.max_episode_length - 1
        done_without_timeouts = done * (1-timeout.float())

        transition_dict = {}
        transition_dict['state_dict'] = copy.deepcopy(curr_state_dict)
        transition_dict['next_state_dict'] = copy.deepcopy(next_state_dict)
        transition_dict['goal_dict'] = reset_data['goal_dict']
        transition_dict['actions'] = copy.deepcopy(actions)
        transition_dict['obs'] = curr_obs.clone()
        transition_dict['next_obs'] = next_obs.clone()
        transition_dict['cost'] = cost
        transition_dict['done'] = done_without_timeouts
        transition_dict['timeout'] = timeout
        
        transition_dict_list.append(transition_dict)
        
        curr_state_dict = copy.deepcopy(next_state_dict)
        curr_obs = next_obs.clone()

        #reset if done
        done_indices = done.nonzero(as_tuple=False).squeeze(-1)
        done_episode_costs = curr_costs[done_indices]
        curr_num_eps_done = len(done_indices)
        curr_num_eps_terminated = torch.sum(done_without_timeouts).item()

        episodes_done += curr_num_eps_done
        episodes_terminated += curr_num_eps_terminated
        curr_num_steps = cost.shape[0]
        total_steps_collected += curr_num_steps

        if curr_num_eps_done > 0:
            #Add done episode to buffer
            episode_dict = cat_dict_list(transition_dict_list, 0)

            if update_buffer:
                buffer.add(episode_dict)
                
            #compute episode metrics
            episode_metrics_list.append(task.compute_metrics(episode_dict))
            episode_cost_buffer.append(curr_costs[0].item())
                                
            #Reset everything
            reset_data = task.reset_idx(done_indices)
            curr_state_dict = envs.reset(reset_data)
            policy.reset(reset_data)
            curr_obs = task.forward(curr_state_dict)[0]
            curr_obs = curr_obs.view(envs.num_envs, obs_dim)
            transition_dict_list = []
            

        #Reset costs and episode_lens for episodes that are done only
        not_done = 1.0 - done.float()
        curr_costs = curr_costs * not_done
        episode_lens = episode_lens * not_done

    if len(episode_cost_buffer) > 0:
        avg_episode_cost = np.average(episode_cost_buffer).item()

    #Consolidate emtrics to be returned        
    metrics = {
        'num_steps_collected': total_steps_collected,
        'num_eps_completed': episodes_done,
        'num_eps_terminated': episodes_terminated,
        'avg_episode_cost': avg_episode_cost,
        }
    
    if buffer is not None:
        metrics['buffer_size'] = len(buffer)

    episode_metrics_keys = episode_metrics_list[0].keys()
    for k in episode_metrics_keys:
        avg_val = np.average([m[k] for m in episode_metrics_list]).item()
        metrics[k] = avg_val

    return buffer, metrics
