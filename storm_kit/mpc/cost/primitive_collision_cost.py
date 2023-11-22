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
from torch.profiler import record_function
from ...geom.sdf.robot_world import RobotWorldCollisionPrimitive

class PrimitiveCollisionCost(nn.Module):
    def __init__(self, weight=None, world_params=None, robot_collision_params=None, world_collision_params=None, batch_size:int=1,
                 distance_threshold=0.1, device:torch.device=torch.device('cpu')):

        super(PrimitiveCollisionCost, self).__init__()
        
        self.device = device
        self.weight = torch.as_tensor(weight, device=self.device)
        self.robot_collision_params = robot_collision_params #robot_params['robot_collision_params']
        self.world_collision_params = world_collision_params
        self.batch_size = batch_size
        bounds = torch.as_tensor(self.world_collision_params['bounds'], device=self.device)
        self.robot_world_coll = RobotWorldCollisionPrimitive(self.robot_collision_params,
                                                             world_params['world_model'],
                                                             robot_batch_size=batch_size,
                                                             device=self.device,
                                                             bounds=bounds,
                                                             grid_resolution=self.world_collision_params['grid_resolution'])
        
        self.n_world_objs = self.robot_world_coll.world_coll.n_objs
        self.t_mat = None
        self.distance_threshold = distance_threshold
    
    def forward(self, link_pos_batch:torch.Tensor, link_rot_batch:torch.Tensor):
        
        inp_device = link_pos_batch.device
        # batch_size = link_pos_batch.shape[0]
        # horizon = link_pos_batch.shape[1]
        # n_links = link_pos_batch.shape[-2]

        # link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        # link_rot_batch = link_rot_batch.view(batch_size * horizon, n_links, 3, 3)
        
        with record_function("primitive_collision_cost:check_sphere_collision"):
            world_coll_dist, self_coll_dist = self.robot_world_coll.check_robot_sphere_collisions(link_pos_batch, link_rot_batch)
        
        #world collision cost
        world_coll_dist = world_coll_dist #.view(batch_size, horizon, n_links)
        #cost only when world_coll_dist is less
        world_coll_dist += self.distance_threshold

        world_coll_dist[world_coll_dist <= 0.0] = 0.0
        world_coll_dist[world_coll_dist > 0.2] = 0.2
        world_coll_dist = world_coll_dist / 0.25
        
        world_cost = torch.sum(world_coll_dist, dim=-1)
        
        #self collision cost
        self_coll_dist = self_coll_dist #.view(batch_size, horizon, n_links)
        # cost only when self_coll_dist is less
        self_coll_dist += self.distance_threshold

        self_coll_dist[self_coll_dist <= 0.0] = 0.0
        self_coll_dist[self_coll_dist > 0.2] = 0.2
        self_coll_dist = self_coll_dist / 0.25
        
        self_cost = torch.sum(self_coll_dist, dim=-1)
        
        cost = world_cost + self_cost
        cost = self.weight * cost 

        return cost.to(inp_device)



