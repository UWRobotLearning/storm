import subprocess
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
eval_script = os.path.join(current_dir, 'evaluate_policy2.py')
vf_agents = ['agent_checkpoint_50ep_ee_state_obs_ensemble_logsumexp.pt']
prediction_temps = [10,20,30,40,50]
success = {agent: {} for agent in vf_agents}
number_of_steps = {agent: {} for agent in vf_agents}

for model_filename in vf_agents:
    for pred_temp in prediction_temps:
        command = [
            'python', eval_script,
            'task=FrankaTrayReacher',
            'eval.num_episodes=1',
            'eval.load_critic=True',
            f'eval.vf_trained_agent={model_filename}',
            'seed=42',
            f'train.vf.prediction_temp={pred_temp}',
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        # import pdb; pdb.set_trace()

        if result.returncode != 0:
            print(f"Error running subprocess: {result.stderr}")
            continue
        
        # extract the last line from stdout which contains the JSON metrics
        stdout_lines = result.stdout.splitlines()
        json_metrics_line = stdout_lines[-1]
        # parse json metrics        
        try:
            metrics = json.loads(json_metrics_line)
            print(f"Metrics for {model_filename} with pred_temp {pred_temp}: {metrics}")
            num_successes = sum(episode['success'] for episode in metrics)
            mean_number_of_steps = sum(episode['number_of_steps'] for episode in metrics) / len(metrics)
            # normalized_number_of_steps = sum(episode['number_of_steps'] for episode in metrics)/
            success[model_filename][pred_temp] = num_successes
            number_of_steps[model_filename][pred_temp] = mean_number_of_steps

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON output: {e}")
            continue

# heatmap
# df_success = pd.DataFrame(success).T
# plt.figure(figsize=(10, 6))
# sns.heatmap(df_success, annot=True, cmap="YlGnBu", cbar=True)
# plt.title('Success Rates Heatmap')
# plt.xlabel('Prediction Temperature')
# plt.ylabel('VF Agent')
# plt.show()

# df_number_of_steps = pd.DataFrame(number_of_steps).T
# plt.figure(figsize=(10, 6))
# sns.heatmap(df_number_of_steps, annot=True, cmap="YlGnBu", cbar=True)
# plt.title('Number of Steps Heatmap')
# plt.xlabel('Prediction Temperature')
# plt.ylabel('VF Agent')
# plt.show()
# Create DataFrame for heatmaps
df_success = pd.DataFrame(success).T
df_number_of_steps = pd.DataFrame(number_of_steps).T

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(df_success, annot=True, cmap="YlGnBu", cbar=True)
plt.title('Success Rates Heatmap')
plt.xlabel('Prediction Temperature')
plt.ylabel('VF Agent')

plt.subplot(1, 2, 2)
sns.heatmap(df_number_of_steps, annot=True, cmap="YlGnBu", cbar=True)
plt.title('Number of Steps Heatmap')
plt.xlabel('Prediction Temperature')
plt.ylabel('VF Agent')

plt.tight_layout()
plt.show()