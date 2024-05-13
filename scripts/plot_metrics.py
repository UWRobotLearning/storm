import subprocess
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
eval_script = os.path.join(current_dir, 'evaluate_policy2.py')
vf_agents = ['agent_checkpoint_50ep_ee_state_obs_ensemble_logsumexp.pt','agent_checkpoint_50ep_ee_state_obs_ensemble_logsumexp_discount_0.pt']
for model_filename in vf_agents:
    command = [
        'python', eval_script,
        'task=FrankaTrayReacher',
        # 'real_robot_exp=False',
        'eval.num_episodes=5',
        'eval.load_critic=True',
        f'eval.vf_trained_agent={model_filename}',
        'seed=42'

        # f'+agent.model_filename={model_filename}'
    ]

    # use subprocess to run the command
    subprocess.run(command)

