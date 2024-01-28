## Install Instructions

### System Dependencies:
- Conda version >= 4.9
- NVIDIA driver >= 460.32
- Cuda toolkit >= 11.8

Steps:

1. Create a new conda environment with: `conda create -n storm_kit_v2 python=3.8`

2. Install PyTorch: https://pytorch.org/

3. Install dependencies `conda env update -f environment.yml`

4. Install python bindings for Isaac Gym Preview 4: https://developer.nvidia.com/isaac-gym

5. Install IsaacGymEnvs: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

6. Run the following command from current directory: `pip install -e . `

### Running Example

1. run scripts/train_self_collision.py to get weights for robot self collision checking.

2. Run python franka_reacher.py, which will launch isaac gym with a franka robot trying to reach a red mug. In the isaac gym gui, search for "ee_target" and toggle "Edit DOF", now you can move the target pose by using the sliders.

