import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from storm_kit.mpc.control.control_utils import generate_halton_samples
from storm_kit.learning.util import fit_mlp
from storm_kit.learning.networks.utils import mlp
from storm_kit.mpc.cost.robot_self_collision_cost import RobotSelfCollisionCost
from storm_kit.differentiable_robot_model import DifferentiableRobotModel


device = torch.device('cuda', 0) 
n_dofs = 7
d_state = 3 * n_dofs
vel_scale = 0.5
urdf_path = '../../../content/assets/urdf/franka_description/franka_panda_no_gripper.urdf'
link_names = ['panda_link1', 'panda_link2','panda_link3','panda_link4','panda_link5', 'panda_link6','panda_hand']
ee_link_name = 'ee_link'
robot_collision_params = {
    'urdf': "urdf/franka_description/franka_panda_no_gripper.urdf",
    'sample_points': 100,
    'link_objs': ['panda_link1', 'panda_link2', 'panda_link3','panda_link4','panda_link5', 'panda_link6','panda_hand'],
    'threshold': 0.35,
    'bounds': [[-0.5, -0.8, 0.0],[0.5,0.8,1.0]],
    'collision_spheres': '../robot/franka.yml',
    'dof': 7,
  }
robot_params = {'robot_collision_params': robot_collision_params}
distance_threshold = 0.05

def generate_dataset(num_data_points, val_ratio=0.2):
    robot_model = DifferentiableRobotModel(urdf_path, device=device)
    self_collision_cost = RobotSelfCollisionCost(weight=1.0, robot_params=robot_params, device=device) 

    joint_lim_dicts = robot_model.get_joint_limits()
    q_pos_uppper = torch.zeros(n_dofs, device=device)
    q_pos_lower = torch.zeros(n_dofs, device=device)
    
    for i in range(n_dofs):
        q_pos_uppper[i] = joint_lim_dicts[i]['upper']
        q_pos_lower[i] = joint_lim_dicts[i]['lower']
    range_b = q_pos_uppper - q_pos_lower

    #Generate data
    q_pos_samples = generate_halton_samples(num_data_points, n_dofs, use_ghalton=True,
                                            device=device)
    #Project samples into state limits
    q_pos_samples = q_pos_samples * range_b + q_pos_lower
    q_pos_samples = q_pos_samples.view(num_data_points, n_dofs)
    q_vel_samples = torch.zeros_like(q_pos_samples)

    ee_pos, ee_rot, lin_jac, ang_jac_seq = robot_model.compute_fk_and_jacobian(
        q_pos_samples, q_vel_samples, link_name= ee_link_name)
    link_pos_seq, link_rot_seq = [], []

    for ki,k in enumerate(link_names):
        link_pos, link_rot = robot_model.get_link_pose(k)
        link_pos_seq.append(link_pos.unsqueeze(1))
        link_rot_seq.append(link_rot.unsqueeze(1))

    link_pos_seq = torch.cat(link_pos_seq, axis=1).unsqueeze(1)
    link_rot_seq = torch.cat(link_rot_seq, axis=1).unsqueeze(1)
    targets = self_collision_cost.distance(link_pos_seq, link_rot_seq)
    
    #compute train-val-split
    num_val_points = int(val_ratio * num_data_points)    
    x_train = q_pos_samples[0:-num_val_points]
    y_train = torch.zeros(x_train.shape[0],1, device=device)
    y_train[targets[:-num_val_points,0]>= -1.0*distance_threshold] = 1.0

    # y_train = targets[0:-num_val_points]
    x_val = q_pos_samples[-num_val_points:]
    y_val = torch.zeros(x_val.shape[0], 1, device=device)
    y_val[targets[-num_val_points:, 0] >= -1.0*distance_threshold] = 1.0

    # y_val = targets[-num_val_points:]
    x_coll_train = x_train[y_train[:,0] > 0.]#.cpu().numpy()
    y_coll_train = y_train[y_train[:,0] > 0.]#.cpu().numpy()
    x_coll_val = x_val[y_val[:,0]> 0.]#.cpu().numpy()
    y_coll_val = y_val[y_val[:,0]> 0.]#.cpu().numpy()

    num_pos_examples = x_coll_train.shape[0]
    num_neg_examples = x_train.shape[0] - num_pos_examples


    # #extract samples that are in collision 
    # x_coll_train = x_train[y_train[:,0]> -1.0*distance_threshold]#.cpu().numpy()
    # y_coll_train = y_train[y_train[:,0]> -1.0*distance_threshold]#.cpu().numpy()
    # x_coll_val = x_val[y_val[:,0]> -1.0*distance_threshold]#.cpu().numpy()
    # y_coll_val = y_val[y_val[:,0]> -1.0*distance_threshold]#.cpu().numpy()

    return x_train, y_train, x_val, y_val, x_coll_train, y_coll_train, x_coll_val, y_coll_val


def train_network():
    num_data_points = 50000
    val_ratio = 0.1
    num_epochs = 100
    batch_size = 128

    mlp_params = {
        'hidden_layers': [256, 256, 256],
        'activation': 'torch.nn.ReLU', 
        'output_activation': None,
        'dropout_prob': 0.0,
        'layer_norm': False,
    }
    in_dim = n_dofs
    out_dim = 1
    layer_sizes = [in_dim] + mlp_params['hidden_layers'] + [out_dim]

    net = mlp(
        layer_sizes=layer_sizes, 
        activation=mlp_params['activation'],
        output_activation=mlp_params['output_activation'],
        dropout_prob=mlp_params['dropout_prob'],
        layer_norm=mlp_params['layer_norm'])

    data = generate_dataset(num_data_points, val_ratio)
    x_train, y_train, x_val, y_val, x_aux_train, y_aux_train, x_aux_val, y_aux_val = data
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    loss_fn = nn.BCEWithLogitsLoss()
    fit_mlp(net, x_train, y_train, x_val, y_val, optimizer, loss_fn, num_epochs, batch_size, device=device)



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    train_network()