import os
import torch
from storm_kit.learning.replay_buffers import RobotBuffer
from storm_kit.learning.learning_utils import buffer_from_folder

# def load_buffer(filepath):
#     print('Loading buffer from {}'.format(filepath))
#     buffer = RobotBuffer(capacity=1000, n_dofs=7)
#     buffer.load(filepath)
#     print('Loaded buffer {}'.format(buffer))
#     return buffer

# def load_episode_data(data_dir):
#     files = sorted(os.listdir(data_dir))
#     #TODO: Sort by episode number!!!!
#     episode_buffers = []
#     for file in files:
#         filepath = os.path.join(data_dir, file)
#         ep_buff = load_buffer(filepath)
#         episode_buffers.append(ep_buff) 

if __name__ == '__main__':
    data_dir = "/home/mohak/workspace/storm/data/simple_reaching_mpc"
    buffer = buffer_from_folder(data_dir)
    print(buffer)