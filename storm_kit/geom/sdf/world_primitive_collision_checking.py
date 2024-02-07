from typing import Dict, List
import torch
import torch.nn as nn
from torch.profiler import record_function



class WorldPrimitiveCollision(nn.Module):
    """Collision checking where the obstacles in the world are 
    represented as spheres or cuboids
    """
    _world_spheres: List[torch.Tensor]
    _world_cubes: List[Dict[str, torch.Tensor]]
    def __init__(self, cfg, world_cfg, device:torch.device = torch.device('cpu')):
        self.cfg = cfg
        self.world_cfg = world_cfg
        self._device = device
        self._world_spheres = []
        self._world_cubes = []
        self.n_objs:int = 0
        self.num_sphere_objs:int = 0
        self.num_cube_objs:int = 0

    def initialize(self):
        pass

    def get_robot_world_sdf(self, robot_link_spheres_dict:Dict[str, torch.Tensor]):
        pass


    def get_pts_sdf(self, pts: torch.Tensor):
        '''
        Given a batch of 3-D points find the signed 
        distance to closest obstacle 
        Args:
        pts: [n,3]
        '''
        #check if points are in bounds
        with record_function('sdf:in_bounds'):
            in_bounds = (pts > self.bounds[0] + self.pitch).all(dim=-1)
            in_bounds &= (pts < self.bounds[1] - self.pitch).all(dim=-1)

        #convert continuous points to voxel indices
        with record_function('sdf:voxel_inds'):
            pt_idx = self.voxel_inds(pts)
        
        pt_idx[~in_bounds] = 0
        
        # remove points outside voxel region:
        # check collisions:
        # batch * n_links, n_pts
        # get sdf from scene voxels:
        # negative distance is outside mesh:
        sdf = self.scene_sdf[pt_idx]
        sdf[~in_bounds] = -10.0
        return sdf