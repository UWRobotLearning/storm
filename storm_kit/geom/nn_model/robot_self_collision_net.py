#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU, ReLU6
# from .network_macros import MLPRegression, scale_to_base, scale_to_net
from ...util_file import get_weights_path, join_path
from storm_kit.learning.networks.utils import mlp
from storm_kit.learning.learning_utils import scale_to_base, scale_to_net

class RobotSelfCollisionNet(nn.Module):
    """This class loads a network to predict the signed distance given a robot joint config."""
    
    def __init__(self, n_joints, norm_dict=None, device=torch.device('cpu')):
        """initialize class

        Args:
            n_joints (int): Number of joints, same as input dimension
        """        
        
        super().__init__()
        self.device = device
        self.norm_dict = norm_dict
        # act_fn = ReLU6
        # in_channels = n_joints
        
        # out_channels = 1
        # dropout_ratio = 0.1
        # mlp_layers = [256, 256, 256]
        self.mlp_params = {
            'hidden_layers': [256, 256, 256],
            'activation': 'torch.nn.ReLU6', 
            'output_activation': None,
            'dropout_prob': 0.1,
            'layer_norm': False,
        }
        self.use_position_encoding = False
        in_dim = 2*n_joints if self.use_position_encoding else n_joints 
        out_dim = 1
        layer_sizes = [in_dim] + self.mlp_params['hidden_layers'] + [out_dim]


        # self.model = MLPRegression(in_channels, out_channels, mlp_layers,
        #                            dropout_ratio, batch_norm=False, act_fn=act_fn,
        #                            layer_norm=False, nerf=True)

        self.net = mlp(
            layer_sizes=layer_sizes, 
            activation=self.mlp_params['activation'],
            output_activation=self.mlp_params['output_activation'],
            dropout_prob=self.mlp_params['dropout_prob'],
            layer_norm=self.mlp_params['layer_norm'])
        
        self.norm_dict = None
    
    def forward(self, q_pos:torch.Tensor):
        if self.norm_dict is not None:
            q_pos = scale_to_net(q_pos, self.norm_dict, 'x')
        input = q_pos
        if self.use_position_encoding:
            input = torch.cat((torch.sin(q_pos), torch.cos(q_pos)), dim=-1)

        score = self.net.forward(input)
        return score
            
    def compute_signed_distance(self, q_pos:torch.Tensor):
        """Compute the signed distance given the joint config.
        (Only to be used when network is being trained for 
        predicting signed distance using regression)

        Args:
            q (tensor): input batch of joint configs [b, n_joints]

        Returns:
            [tensor]: largest signed distance between any two non-consecutive links of the robot.
        """        
        with torch.no_grad():
            # q_scale = scale_to_net(q, self.norm_dict,'x')
            # dist = self.model.forward(q_scale)
            # dist_scale = scale_to_base(dist, self.norm_dict, 'y')
            dist = self.forward(q_pos)
            dist_scale = scale_to_base(dist, self.norm_dict, 'y')
        return dist_scale

    def get_collision_prob(self, q_pos:torch.Tensor):
        """Check collision given joint config. 
        (Requires classifier like training.)

        Args:
            q (tensor): input batch of joint configs [b, n_joints]

        Returns:
            [tensor]: probability of collision of links, from sigmoid value.
        """        
        with torch.no_grad():
            # q_scale = scale_to_net(q, self.norm_dict,'x')
            # dist = torch.sigmoid(self.model.forward(q_scale))
            score = self.forward(q_pos)
            prob = torch.sigmoid(score)
        return prob


    def load_parameters(self, f_name):
        """Loads pretrained network weights if available.

        Args:
            f_name (str): file name, this is relative to weights folder in this repo.
            tensor_args (Dict): device and dtype for pytorch tensors
        """        
        try:
            chk = torch.load(join_path(get_weights_path(), f_name))
            self.load_state_dict(chk["model_state_dict"])
            if "norm" in chk:
                self.norm_dict = chk["norm"]
                for k in self.norm_dict.keys():
                    self.norm_dict[k]['mean'] = self.norm_dict[k]['mean'].to(self.device)
                    self.norm_dict[k]['std'] = self.norm_dict[k]['std'].to(self.device)
            weights_loaded = True
        except Exception as e:
            print(e)
            
            weights_loaded = False
            print('WARNING: Weights not loaded')
        
        self.net = self.net.to(self.device)
        self.net.eval()
        return weights_loaded