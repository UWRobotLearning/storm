import csv, json, random, string, sys
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import time



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


def fit_mlp(
    net, 
    x_train, y_train,
    x_val, y_val, 
    optimizer, loss_fn, 
    num_epochs, batch_size, 
    x_train_aux=None, y_train_aux=None,
    x_val_aux=None, y_val_aux=None,
    device=torch.device('cpu')):

    net.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    pbar = tqdm(range(int(num_epochs)) , unit="epoch", mininterval=0, disable=False, desc='train')
    num_batches = x_train.shape[0] // batch_size #throw away last incomplete batch

    for i in pbar:
        #random permutation of data
        rand_idxs = torch.randperm(x_train.shape[0])
        avg_loss = 0.0
        avg_acc = 0.0
        
        for batch_num in range(0, num_batches):
            batch_idxs = rand_idxs[batch_num*batch_size: (batch_num+1)*batch_size]
            
            x_batch = x_train[batch_idxs].to(device)
            y_batch = y_train[batch_idxs].to(device)

            y_pred = net.forward(x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (torch.sigmoid(y_pred).round() == y_batch).float().mean().cpu()
            avg_loss += loss.item()
            avg_acc += acc.item()
        avg_loss /= num_batches*1.0
        avg_acc /= num_batches*1.0
        #run validation
        net.eval()
        with torch.no_grad():
            y_pred_val = net.forward(x_val)
            loss_val = loss_fn(y_pred_val, y_val)
            acc_val = (torch.sigmoid(y_pred_val).round() == y_val).float().mean().cpu()
        net.train()

        pbar.set_postfix(
            avg_loss_train=float(avg_loss),
            avg_acc_train=float(avg_acc),
            avg_loss_val = float(loss_val.item()),
            avg_acc_val = float(acc_val.item())
        )