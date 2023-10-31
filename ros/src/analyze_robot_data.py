import argparse
import pickle
import matplotlib.pyplot as plot
import os

def plot_data(filename):
    root_dir = '../../robot_data'
    filepath = os.path.abspath(os.path.join(root_dir, filename + '.pkl'))
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(data.keys())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='DataAnalyzer',
                        description='Plot robot data from pickle file')
    parser.add_argument('--filename', type=str)
    args=parser.parse_args()
    plot_data(args.filename)
