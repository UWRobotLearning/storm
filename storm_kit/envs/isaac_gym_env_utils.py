#@mohak: add utilities for loading robot, table, objects etc. to avoid code repetition
import os   
from isaacgym import gymapi
import torch

def tensor_clamp(t, min, max):
    t = torch.where(t > min, t, min)
    t = torch.where(t < max, t, max)
    return t

def load_primitive_asset(gym_instance, sim_instance, asset_dims:gymapi.Vec3, asset_type: str= 'box', fix_base_link:bool=False, disable_gravity:bool=False):
    asset_options = gymapi.AssetOptions()
    # asset_options.armature = 0.001
    asset_options.fix_base_link = fix_base_link
    asset_options.disable_gravity = disable_gravity
    # asset_options.thickness = 0.002
    if asset_type == 'box':
        asset = gym_instance.create_box(sim_instance, asset_dims.x, asset_dims.y, asset_dims.z, asset_options)
    return asset #, table_dims, table_color


def load_urdf_asset(gym_instance, sim_instance, asset_file:str, fix_base_link:bool=False, disable_gravity:bool=False):
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../content/assets")
    asset_options = gymapi.AssetOptions()
    asset_options.flip_visual_attachments = True
    asset_options.fix_base_link = fix_base_link
    asset_options.collapse_fixed_joints = False #True #for parsing fixed joints from the urdf as well
    asset_options.disable_gravity = disable_gravity
    asset_options.thickness = 0.001
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    asset_options.use_mesh_materials = True
    asset = gym_instance.load_asset(sim_instance, asset_root, asset_file, asset_options)
    return asset  