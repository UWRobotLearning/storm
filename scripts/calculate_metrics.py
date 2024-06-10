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
    [75, 85, 13.16, 0.60, 1.13],  # Trial 1
    [75, 90, 14.20, 0.57, 1.10],  # Trial 2
    [70, 80, 15.32, 0.65, 1.09]  # Trial 3
]

statistics = calculate_statistics(trial_data)

for metric, stats in statistics.items():
    print(f"{metric.capitalize()}: {stats['mean']:.2f} ± {stats['std_error']:.2f}")
