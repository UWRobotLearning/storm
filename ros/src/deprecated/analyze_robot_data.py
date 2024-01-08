import argparse
import pickle
import matplotlib.pyplot as plt
import os
from storm_kit.util_file import get_root_path

ROOT_DIR = os.path.join(get_root_path(), 'robot_data')

JOINT_IDXS_TO_PLOT = [0, 1, 2, 3, 4, 5, 6]

def plot_data(filename):
    filepath = os.path.join(ROOT_DIR, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    fig, ax = plt.subplots(3,1)
    # for key in data.keys():
    for i in range(data['q_vel'].shape[1]):
        ax[0].plot(data['q_vel'][:,i])
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='DataAnalyzer',
                        description='Plot robot data from pickle file')
    parser.add_argument('--filename', type=str)
    args=parser.parse_args()
    plot_data(args.filename)
