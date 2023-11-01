import pickle
import matplotlib.pyplot as plt
import numpy as np

# def empty_list():
#     return []
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
filename = "/home/navneet/catkin_ws/src/storm/robot_data/robot_data_3_Oct30.pkl"
data = load_data(filename)
q_pos_array = np.array(data['q_pos'])

# For example, if you want to plot 'q_pos'
plt.figure(figsize=(10, 5))
plt.plot(q_pos_array[:, 0, 0])  # This will plot the first element of the last dimension
# plt.show()
plt.title('q_pos over time')
plt.xlabel('Time')
plt.ylabel('q_pos')
plt.show()