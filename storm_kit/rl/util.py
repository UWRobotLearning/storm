import csv, json, random, string, sys
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import torch



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