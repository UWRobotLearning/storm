import subprocess
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

#extract ensemble sizes from model filenames
def get_ensemble_size(filename):
    match = re.search(r'ensemble_(\d+)', filename)
    return int(match.group(1)) if match else None

#load metrics from file if it exists
def load_metrics(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

#save metrics to file
def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f)

#for plotting, format numbers as integers if they are greater than 100 or if they are integers
def custom_formatter(x):
    if isinstance(x, int) or x.is_integer():
        return f"{int(x)}"
    else:
        return f"{x:.2f}"

current_dir = os.path.dirname(os.path.realpath(__file__))
eval_script = os.path.join(current_dir, 'evaluate_policy2.py')
metrics_folder = os.path.join(current_dir, 'metrics')
if not os.path.exists(metrics_folder):
    os.makedirs(metrics_folder)
metrics_file = os.path.join(metrics_folder, 'metrics_temp.json')

vf_agents = ['agent_checkpoint_50ep_ee_obs_real_robot_cube_center_may29_ensemble_100.pt',]
            #  'agent_checkpoint_50ep_ee_all_obs_may19_ensemble_20.pt',
            #  'agent_checkpoint_50ep_ee_all_obs_may19_ensemble_40.pt', 'agent_checkpoint_50ep_ee_all_obs_may19_ensemble_60.pt',
            #  'agent_checkpoint_50ep_ee_all_obs_may19_ensemble_80.pt', 'agent_checkpoint_50ep_ee_all_obs_may19_ensemble_100.pt',]
prediction_temps = [20,]#10,20,30,40,50]
# success = {agent: {} for agent in vf_agents}
# number_of_steps = {agent: {} for agent in vf_agents}

metrics_dict = {str(get_ensemble_size(agent)): {} for agent in vf_agents}
success = {str(get_ensemble_size(agent)): {} for agent in vf_agents}
number_of_steps = {str(get_ensemble_size(agent)): {} for agent in vf_agents}
number_of_friction_cone_violations = {str(get_ensemble_size(agent)): {} for agent in vf_agents}
ee_dist_error = {str(get_ensemble_size(agent)): {} for agent in vf_agents}
rotation_error = {str(get_ensemble_size(agent)): {} for agent in vf_agents}
abs_cube_pos_change = {str(get_ensemble_size(agent)): {} for agent in vf_agents}

loaded_metrics = load_metrics(metrics_file)
load_metrics = False

if not load_metrics:
    for model_filename in vf_agents:
        ensemble_size = get_ensemble_size(model_filename)
        print(ensemble_size)
        for pred_temp in prediction_temps:
            command = [
                'python', eval_script,
                'task=FrankaTrayReacherRealRobot',
                'eval.num_episodes=20',
                'eval.load_critic=True',
                f'eval.vf_trained_agent={model_filename}',
                'seed=42', #42 is the default seed, for strict eval --> change seed
                f'train.vf.prediction_temp={pred_temp}',
                f'train.vf.ensemble_size={ensemble_size}',
                'real_robot_exp=True'
            ]

            result = subprocess.run(command, capture_output=True, text=True)
            # import pdb; pdb.set_trace()

            if result.returncode != 0:
                print(f"Error running subprocess: {result.stderr}")
                continue
            
            #extract the last line from stdout which contains the JSON metrics
            stdout_lines = result.stdout.splitlines()
            json_metrics_line = stdout_lines[-1]
            #parse json metrics        
            try:
                metrics = json.loads(json_metrics_line)
                print(f"Metrics for {model_filename} with pred_temp {pred_temp}: {metrics}")
                metrics_dict[str(ensemble_size)][pred_temp] = metrics
                filtered_episodes = [ep for ep in metrics if ep['abs_cube_pos_change'] < 0.01]
                num_successes = sum(episode['success'] for episode in filtered_episodes)
                mean_number_of_steps = sum(episode['number_of_steps'] for episode in metrics) / len(metrics)
                ee_dist_error[str(ensemble_size)][int(pred_temp)] = sum(episode['dist_err_final'] for episode in metrics) / len(metrics)
                rotation_error[str(ensemble_size)][int(pred_temp)] = sum(episode['rot_err_final'] for episode in metrics) / len(metrics)
                abs_cube_pos_change[str(ensemble_size)][int(pred_temp)] = sum(episode['abs_cube_pos_change'] for episode in metrics) / len(metrics)
                number_of_friction_cone_violations[str(ensemble_size)][int(pred_temp)] = sum(episode['number_of_friction_cone_violations'] for episode in metrics) / len(metrics)  

                # normalized_number_of_steps = sum(episode['number_of_steps'] for episode in metrics)/
                success[str(ensemble_size)][pred_temp] = num_successes
                number_of_steps[str(ensemble_size)][pred_temp] = mean_number_of_steps

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON output: {e}")
                continue
    save_metrics(metrics_dict, metrics_file)
#implement how to extract data from loaded_metrics
else:
    for ensemble_size, temps in loaded_metrics.items():
        for pred_temp, metrics in temps.items():
            num_successes = sum(episode['success'] for episode in metrics)
            mean_number_of_steps = sum(episode['number_of_steps'] for episode in metrics) / len(metrics)
            success[ensemble_size][int(pred_temp)] = num_successes
            number_of_steps[ensemble_size][int(pred_temp)] = mean_number_of_steps
            number_of_friction_cone_violations[ensemble_size][int(pred_temp)] = sum(episode['number_of_friction_cone_violations'] for episode in metrics) / len(metrics)  
            ee_dist_error[ensemble_size][int(pred_temp)] = sum(episode['dist_err_final'] for episode in metrics) / len(metrics)
            rotation_error[ensemble_size][int(pred_temp)] = sum(episode['rot_err_final'] for episode in metrics) / len(metrics)
            print("success", num_successes)
            
#heatmap
plot=False
if plot:
    df_success = pd.DataFrame(success).T
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df_success, annot=True, cmap="YlGnBu", cbar=True, annot_kws={"size": 10})
    for text in ax.texts:
            text.set_text(custom_formatter(float(text.get_text())))
    plt.title('Number of successful episodes')
    plt.xlabel('Prediction Temperature')
    plt.ylabel('Ensemble Size')
    plt.savefig('plots/success_rates_heatmap_only_friction_cost_train_ensemble_100_seed_42.png')
    plt.show() 

    df_number_of_steps = pd.DataFrame(number_of_steps).T
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df_number_of_steps, annot=True, fmt='g', cmap="YlGnBu", cbar=True, annot_kws={"size": 10})
    for text in ax.texts:
            text.set_text(custom_formatter(float(text.get_text())))
    plt.title('Number of Steps')
    plt.xlabel('Prediction Temperature')
    plt.ylabel('Ensemble Size')
    plt.savefig('plots/number_of_steps_heatmap_only_friction_cost_train_ensemble_100_seed_42.png')
    plt.show()

    df_number_of_friciton_cone_violations = pd.DataFrame(number_of_friction_cone_violations).T
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df_number_of_friciton_cone_violations, fmt='g', annot=True, cmap="YlGnBu", cbar=True, annot_kws={"size": 10})
    for text in ax.texts:
            text.set_text(custom_formatter(float(text.get_text())))
    plt.title('Number of Friction Cone Violations')
    plt.xlabel('Prediction Temperature')
    plt.ylabel('Ensemble Size')
    plt.savefig('plots/number_of_friction_cone_violations_heatmap_only_friction_cost_train_ensemble_100_seed_42.png')
    plt.show()

    df_ee_dist_error = pd.DataFrame(ee_dist_error).T
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df_ee_dist_error, annot=True, cmap="YlGnBu", fmt='g', cbar=True, annot_kws={"size": 10})
    for text in ax.texts:
            text.set_text(custom_formatter(float(text.get_text())))
    plt.title('Error between final and target end-effector position (cm)')
    plt.xlabel('Prediction Temperature')
    plt.ylabel('Ensemble Size')
    plt.savefig('plots/ee_dist_error_heatmap_only_friction_cost_train_ensemble_100_seed_42.png')
    plt.show()

    df_rotation_error = pd.DataFrame(rotation_error).T
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df_rotation_error, annot=True, cmap="YlGnBu", fmt='g', cbar=True, annot_kws={"size": 10})
    for text in ax.texts:
            text.set_text(custom_formatter(float(text.get_text())))
    plt.title('Error between final and target end-effector orientation (degrees)')
    plt.xlabel('Prediction Temperature')
    plt.ylabel('Ensemble Size')
    plt.savefig('plots/rotation_error_heatmap_only_friction_cost_train_ensemble_100_seed_42.png')
    plt.show()

    df_abs_cube_pos_change = pd.DataFrame(abs_cube_pos_change).T
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df_abs_cube_pos_change, annot=True, cmap="YlGnBu", fmt='g', cbar=True, annot_kws={"size": 10})
    for text in ax.texts:
            text.set_text(custom_formatter(float(text.get_text())))
    plt.title('Change in absolute cube position (cm)')
    plt.xlabel('Prediction Temperature')
    plt.ylabel('Ensemble Size')
    plt.savefig('plots/abs_cube_pos_change_heatmap_only_friction_cost_train_ensemble_100_seed_42.png')
    plt.show()