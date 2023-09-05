#@mohak: add utilities for loading robot, table, objects etc. to avoid code repetition
import os   
from isaacgym import gymapi
import torch

def tensor_clamp(t, min, max):
    t = torch.where(t > min, t, min)
    t = torch.where(t < max, t, max)
    return t



def load_object_asset(disable_gravity=False):
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../content/assets")
    object_asset_file = "urdf/ball.urdf"

    # if "asset" in self.cfg["env"]:
    #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
    #     object_asset_file = self.cfg["env"]["asset"].get("assetFileNameObject", object_asset_file )
    
    # load object asset
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.collapse_fixed_joints = False
    asset_options.disable_gravity = disable_gravity
    asset_options.thickness = 0.001
    asset_options.use_mesh_materials = True
    object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)

    # self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
    # self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
    # print("num object bodies: ", self.num_object_bodies)
    object_color = gymapi.Vec3(0.0, 0.0, 1.0)

    return object_asset, object_color  