import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from storm_kit.mpc.control.control_utils import generate_halton_samples
from storm_kit.learning.learning_utils import fit_mlp
# from storm_kit.learning.networks.utils import mlp
from storm_kit.geom.nn_model.robot_self_collision_net import RobotSelfCollisionNet
from storm_kit.mpc.cost.robot_self_collision_cost import RobotSelfCollisionCost
from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.util_file import join_path, get_weights_path, get_configs_path, get_assets_path


device = torch.device('cuda', 0) 

def generate_dataset(
        config,
        num_data_points, 
        val_ratio=0.2):
    
    print('Generating self collision data')
    robot_collision_config = config.robot_collision_params
    ee_link_name = config.ee_link_name
    n_dofs = robot_collision_config.n_dofs
    urdf_path = join_path(get_assets_path(), robot_collision_config['urdf_path'])
    link_names = robot_collision_config['link_names']
    distance_threshold = 0.02


    robot_model = DifferentiableRobotModel(urdf_path, device=device)
    self_collision_cost = RobotSelfCollisionCost(weight=1.0, config=robot_collision_config, device=device) 

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

    _,_,_,_ = robot_model.compute_fk_and_jacobian(
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
    y_train = torch.zeros(x_train.shape[0], 1, device=device)
    y_train[targets[:-num_val_points,0]>= -1.0*distance_threshold] = 1.0

    # y_train = targets[0:-num_val_points]
    x_val = q_pos_samples[-num_val_points:]
    y_val = torch.zeros(x_val.shape[0], 1, device=device)
    y_val[targets[-num_val_points:, 0] >= -1.0*distance_threshold] = 1.0

    # y_val = targets[-num_val_points:]
    x_coll_train = x_train[y_train[:,0] > 0.].clone() #.cpu().numpy()
    y_coll_train = y_train[y_train[:,0] > 0.].clone()#.cpu().numpy()
    x_coll_val = x_val[y_val[:,0]> 0.]#.cpu().numpy()
    y_coll_val = y_val[y_val[:,0]> 0.]#.cpu().numpy()

    num_pos_examples = x_coll_train.shape[0]
    num_neg_examples = x_train.shape[0] - num_pos_examples
    print('Num pos examples = {}, Num neg examples = {}'.format(num_pos_examples, num_neg_examples))
    #calculate pos_weight to balance pos and neg examples
    pos_weight = num_neg_examples / num_pos_examples 

    # #extract samples that are in collision 
    # x_coll_train = x_train[y_train[:,0]> -1.0*distance_threshold]#.cpu().numpy()
    # y_coll_train = y_train[y_train[:,0]> -1.0*distance_threshold]#.cpu().numpy()
    # x_coll_val = x_val[y_val[:,0]> -1.0*distance_threshold]#.cpu().numpy()
    # y_coll_val = y_val[y_val[:,0]> -1.0*distance_threshold]#.cpu().numpy()

    return x_train, y_train, x_val, y_val, x_coll_train, y_coll_train, x_coll_val, y_coll_val, pos_weight

@hydra.main(config_name="config", config_path=get_configs_path()+"/gym")
def train_network(cfg:DictConfig):
    checkpoint_dir = get_weights_path() + "/robot_self"
    robot_name = cfg.task.rollout.model.name
    plot = True
    num_data_points = 500000
    val_ratio = 0.1
    num_epochs = 200
    batch_size = 256
    lr = 1e-3
    n_dofs = cfg.task.rollout.n_dofs

    data = generate_dataset(cfg.task.rollout.model, num_data_points, val_ratio)
    x_train, y_train, x_val, y_val, x_train_aux, y_train_aux, x_aux_val, y_aux_val, pos_weight = data

    # net = mlp(
    #     layer_sizes=layer_sizes, 
    #     activation=mlp_params['activation'],
    #     output_activation=mlp_params['output_activation'],
    #     dropout_prob=mlp_params['dropout_prob'],
    #     layer_norm=mlp_params['layer_norm'])

    net = RobotSelfCollisionNet(
        n_joints = n_dofs,
        norm_dict = None,
        use_position_encoding=True,
        device = device
    )

    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    best_net, train_losses, train_accuracies, validation_losses, validation_accuracies, norm_dict = \
        fit_mlp(net, x_train, y_train, x_val, y_val,optimizer, loss_fn, num_epochs, batch_size,  x_train_aux= x_train_aux, y_train_aux=y_train_aux, normalize=True, device=device)
    
    #Save the best found network weights
    model_path = join_path(checkpoint_dir, robot_name +'_self_collision_weights.pt')
    model_dict = {
        'model_state_dict': best_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }
    if norm_dict is not None:
        model_dict['norm'] = norm_dict
   
    print('saving model to {}'.format(model_path))
    torch.save(model_dict, model_path)
    print('model saved')
    
    if plot:
        import matplotlib.pyplot as plt 
        fig, ax = plt.subplots(2,1)
        ax[0].plot(train_losses, label='train')
        ax[0].plot(validation_losses, label='validation')
        ax[1].plot(train_accuracies, label='train')
        ax[1].plot(validation_accuracies, label='validation')
        ax[0].legend()
        ax[0].set_title('Loss')
        ax[1].set_title('Accuracy')
        plt.show()

    

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    train_network()