import numpy as np

def calculate_statistics(trial_data):
    """
    Calculate mean and standard deviation for given trial data.

    Parameters:
    trial_data (list of lists): A list containing three lists of metrics for each trial.

    Returns:
    dict: A dictionary with mean and standard error for each metric.
    """
    metrics = ['success_percent hard', 'success_percent_soft' ,'angle of tilt', 'ee lin vel norm', 'ee ang vel norm']
    stats = {}
    
    for i, metric in enumerate(metrics):
        values = [trial[i] for trial in trial_data]
        mean = np.mean(values)
        std_dev = np.std(values)
        std_error = std_dev / np.sqrt(len(values))
        stats[metric] = {'mean': mean, 'std_dev': std_dev, 'std_error': std_error}
    
    return stats

trial_data = [
    [40, 40, 9.09, 0.51, 1.00],  # Trial 1
    [50, 50, 7.69, 0.40, 0.45],  # Trial 2
    [50, 50, 8.12, 0.35, 0.50]  # Trial 3
]

statistics = calculate_statistics(trial_data)

for metric, stats in statistics.items():
    print(f"{metric.capitalize()}: {stats['mean']:.2f} ± {stats['std_error']:.2f}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
algorithms = ['PWP','ISP']
metrics = ['S1', 'S2', 'Alpha', 'v', 'omega']

# Mean and standard error values for each algorithm and metric
mean_values = {
    'S1': [46.67, 100],
    'S2': [46.47, 100],
    'Alpha': [8.30, 14.48],
    'v': [0.42, 0.63],
    'omega': [0.45, 1.00]
}

std_errors = {
    'S1': [2.72, 0.0],
    'S2': [2.72, 0.0],
    'Alpha': [0.34, 0.38],
    'v': [0.04, 0.04],
    'omega': [0.14, 0.04]
}

# Set seaborn style
sns.set(style="whitegrid")

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(12, 3))

# Colors for each metric
colors = sns.color_palette("husl", 5)
colors = ['#7FC97F', '#BEAED4','#FDC086','#A6CEE3']
X_axis = np.arange(len(algorithms))

# Plot for S1 and S2
axs[0].bar(X_axis - 0.24, mean_values['S1'], yerr=std_errors['S1'], capsize=5, color=colors[0], label="S1 (%)", width=0.48)
axs[0].bar(X_axis + 0.24, mean_values['S2'], yerr=std_errors['S2'], capsize=5, color=colors[1], label="S2 (%)", width=0.48)
axs[0].set_ylabel('Value', fontsize=20)
axs[0].set_ylim(0, 120)
axs[0].set_yticks(np.arange(0, 101, 25))
axs[0].legend(loc='upper right', fontsize=12)

# Plot for Alpha
axs[1].bar(X_axis, mean_values['Alpha'], yerr=std_errors['Alpha'], color=colors[2], capsize=5, label=r'$\alpha_{\mathrm{ee}, t}^{\max} \; (deg)$')
axs[1].set_ylim(0, 18.0)
axs[1].set_yticks(np.arange(0, 17, 4))
axs[1].legend(loc='upper right', fontsize=12)

# Plot for v and omega
axs[2].bar(X_axis - 0.24, mean_values['v'], yerr=std_errors['v'], color=colors[3], capsize=5, width=0.48, label=r'$\|\mathbf{v}_{\mathrm{ee}, t}\|_{\max}\; (m/s)$')
axs[2].bar(X_axis + 0.24, mean_values['omega'], yerr=std_errors['omega'], color=colors[0], capsize=5, width=0.48, label=r'$\|\mathbf{\omega}_{\mathrm{ee}, t}\|_{\max}\; (rad/s)$')
axs[2].set_ylim(0, 1.4)
axs[2].set_yticks(np.arange(0, 1.3, 0.4))
axs[2].legend(loc='upper right', fontsize=12)

# Set common labels
for ax in axs.flat:
    ax.set_xticks(X_axis)
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.show()
