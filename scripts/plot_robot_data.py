import pickle
import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Loading data
filename = "/home/navneet/catkin_ws/src/storm/robot_data/robot_data_3_Oct30.pkl"
data = load_data(filename)

# For example, if you want to plot 'q_pos'
plt.figure(figsize=(10, 5))
plt.plot(data['q_pos'])
plt.title('q_pos over time')
plt.xlabel('Time')
plt.ylabel('q_pos')
plt.show()