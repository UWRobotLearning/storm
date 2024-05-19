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

for model_filename in vf_agents:
    for pred_temp in prediction_temps:
        command = [
            'python', eval_script,
            'task=FrankaTrayReacher',
            'eval.num_episodes=20',
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
            success[model_filename][pred_temp] = num_successes

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON output: {e}")
            continue

# heatmap
df = pd.DataFrame(success).T
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True)
plt.title('Success Rates Heatmap')
plt.xlabel('Prediction Temperature')
plt.ylabel('VF Agent')
plt.show()
