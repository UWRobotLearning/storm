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
import torch.nn as nn
# import torch.nn.functional as F

from ...differentiable_robot_model.coordinate_transform import CoordinateTransform, quaternion_to_matrix

from ...util_file import get_assets_path, join_path
from ...geom.sdf.robot import RobotSphereCollision
from .gaussian_projection import GaussianProjection
from storm_kit.geom.nn_model.robot_self_collision_net import RobotSelfCollisionNet

class RobotSelfCollisionCost(nn.Module):
    def __init__(self, weight, config=None,
                 gaussian_params={}, distance_threshold=-0.01, 
                 batch_size=2, device=torch.device('cpu')):
                #  tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(RobotSelfCollisionCost, self).__init__()
        # self.tensor_args = tensor_args
        self.device = device # tensor_args['device']
        self.config = config
        # self.float_dtype = tensor_args['dtype']
        self.tensor_args={'device':self.device, 'dtype':torch.float32}
        self.distance_threshold = distance_threshold
        self.weight = torch.as_tensor(weight, device=self.device)
        
        # self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)


        # load robot model:
        # robot_collision_params = robot_params['robot_collision_params']
        # robot_collision_params['urdf'] = join_path(get_assets_path(),
        #                                            robot_collision_params['urdf'])


        # load nn params:
        # label_map = robot_params['world_collision_params']['label_map']
        # bounds = robot_params['world_collision_params']['bounds']
        self.distance_threshold = distance_threshold
        self.batch_size = batch_size
        
        # Initialize collision model. This can be used if the NN is not trained.
        # TODO: Add flag to use this model is net weights are not loaded  
        self.collision_model = RobotSphereCollision(self.config, self.batch_size, device=self.device)
        self.collision_model.build_batch_features(batch_size=self.batch_size, clone_pose=True, clone_objs=True)

        self.nn_collision_model = RobotSelfCollisionNet(
            n_joints=self.config['n_dofs'],
            norm_dict=None,
            device=self.device
        )
        self.weights_loaded = False
        try:
            self.weights_loaded = self.nn_collision_model.load_parameters(self.config['self_collision_weights'])
        except:
            pass
        self.res = None
        self.t_mat = None

    def distance(self, link_pos_seq, link_rot_seq):
        """
            Uses analytical model for calculating signed distance

        """
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]
        link_pos = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.collision_model.build_batch_features(batch_size=self.batch_size * horizon, clone_pose=True, clone_objs=True)
        
        res = self.collision_model.check_self_collisions(link_pos, link_rot)
        self.res = res
        res = res.view(batch_size, horizon, n_links)
        res = torch.max(res, dim=-1)[0]
        return res

    def forward(self, q_pos):
        """
            Calculate the collision cost
        """
        batch_size = q_pos.shape[0]
        horizon = q_pos.shape[1]
        q_pos = q_pos.view(batch_size * horizon, q_pos.shape[2])
        
        # res = self.coll.check_self_collisions_nn(q)
        # res = self.coll.check_self_collisions_nn(q)
        if self.weights_loaded:
            res = self.nn_collision_model.get_collision_prob(q_pos)
            res = res.view(batch_size, horizon)
        else:
            res = self.collision_model.check_self_collisions(q_pos)
            res = res.view(batch_size, horizon)
            res += self.distance_threshold
            res[res <= 0.0] = 0.0
            res[res >= 0.5] = 0.5
            # rescale:
            res = res / 0.25

        cost = res
        # cost = self.weight * self.proj_gaussian(cost)
        cost = self.weight * cost

        return cost
    
