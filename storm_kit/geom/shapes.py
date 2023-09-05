import torch
from storm_kit.differentiable_robot_model.coordinate_transform import CoordinateTransform

class Shape():
    def __init__(self) -> None:
        pass

    def compute_mass_data(self, material):
        raise NotImplementedError

    def volume(self):
        raise NotImplementedError

class Sphere(Shape):
    def __init__(self, 
            pose:CoordinateTransform,
            # velocity:torch.tensor, 
            radius:torch.Tensor,
            device: torch.device = torch.device('cpu')):
            # mass: torch.tensor,
            # restitution:torch.tensor):
        
        super().__init__()
        self.pose = pose
        self.radius = radius
        self.device = device
        # self.mass = mass
        # self.restitution = restitution
        # if self.mass == 0.:
        #     self.inv_mass = 0.0
        # else:
        #     self.inv_mass = 1.0 / self.mass
        # assert self.position.device == self.radius.device
    
    def set_pose(self, pose:CoordinateTransform):
        self.pose = pose
    
    def compute_mass_data(self, material):
        density = material.density
        mass = density * self.volume()
        if mass == 0.0:
            inv_mass = 0.0
        else:
            inv_mass = 1.0 / mass
        #TODO: Implement inertia
        inertia = 0.0
        inv_inertia = 0.0
        mass_data = {
            'mass': mass,
            'inv_mass': inv_mass,
            'inertia': inertia,
            'inv_invertia': inv_inertia
        }
        return mass_data
    
    def volume(self):
        return (4/3) * torch.pi * (self.radius ** 3)
    
    def __repr__(self):
        return 'Sphere(radius={})'.format(self.radius)


# class Capsule(Shape):

# class Cuboid(Shape):

# class Cylinder(Shape):