#!/usr/bin/env python
from dataclasses import dataclass
from storm_kit.geom.shapes import Sphere
# from storm_kit.geom.sdf.primitives_new import sphere_sphere_collision
from storm_kit.differentiable_robot_model.coordinate_transform import CoordinateTransform
import torch
import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

@dataclass
class Material:
    density: float
    restitution: float
    mu_static: float
    mu_dynamic: float

# @dataclass
# class MassData:
#     mass: float
#     inv_mass: float
#     inertia: float
#     inverse_inertia: float


class RigidBody():
    def __init__(
            self,
            shape,
            pose,
            material,
            velocity,
            force,
            mass_data=None,
            ):
        
        self.shape = shape
        self.pose = pose
        self.material = material
        if mass_data is None:
            self.mass_data = self.shape.compute_mass_data(self.material)
        else:  self.mass_data = mass_data
        self.velocity = velocity
        self.force = force
        # self.gravity_scale = gravity_scale

    def check_collision(self, query_body):
        coll_data = sphere_sphere_collision(self.shape, query_body.shape)
        #compute tangent plane
        normal = coll_data['normal']
        v_rel = query_body.velocity - self.velocity
        v_rel_normal = torch.sum(v_rel * normal, dim=-1)
        # v_rel_normal_vec = v_rel_normal * normal
        tangent = v_rel - v_rel_normal @ normal
        tangent_norm = torch.norm(tangent, dim=-1).unsqueeze(-1)
        tangent = torch.where(tangent_norm > 0, torch.div(tangent, tangent_norm), tangent)
        v_rel_tangent =  torch.sum(v_rel * tangent, dim=-1) 
        # v_rel_tangent = torch.norm(v_rel_tangent_vec, dim=-1)
        # tangent = v_rel_tangent_vec / v_rel_tangent
        
        coll_data['v_rel_normal'] = v_rel_normal
        coll_data['v_rel_tangent'] = v_rel_tangent
        coll_data['tangent'] = tangent
        
        return coll_data

    def reset_force(self):
        self.force = torch.zeros_like(self.force)

    def update_pose(self, dt):
        new_translation = self.pose.translation() + self.velocity * dt
        self.pose.set_translation(new_translation) 
        self.shape.pose.set_translation(new_translation)
    
    def update_position_delta(self, correction):
        new_translation = self.pose.translation() + correction
        self.pose.set_translation(new_translation) 
        self.shape.pose.set_translation(new_translation)
    
    def set_state(self):
        self.pose.set_translation()
        


class Sim():
    def __init__(self):
        self.vis_initialized = False
        self.vis = None

        bodyA_transform = CoordinateTransform(trans=torch.tensor([[0.0, 0.0, 0.0]]))
        bodyB_transform = CoordinateTransform(trans=torch.tensor([[0.3, 0.2, 0.1]]))
        bodyB_init_velocity = torch.tensor([[-0.5, -0.1, -0.1]])

        A = RigidBody(
            shape=Sphere(
                pose=bodyA_transform,
                radius=torch.tensor([0.1])),
            pose=bodyA_transform,
            material=Material(density=1.0, restitution=1.0, mu_static=1.0, mu_dynamic=0.1),
            velocity=torch.zeros(1,3),
            force=torch.zeros(1,3)
        )
        B = RigidBody(
            shape=Sphere(
                pose = bodyB_transform,
                radius=torch.tensor([0.1])),
            pose=bodyB_transform,
            material=Material(density=1.0, restitution=1., mu_static=1.0, mu_dynamic=0.1),
            velocity=bodyB_init_velocity,
            force=torch.zeros(1,3)
        )

        self.bodies = {'A': A, 'B': B}
    

    def step(self):
        bodyA = self.bodies['A']
        bodyB = self.bodies['B']
        # coll_data = sphere_sphere_collision(bodyA.shape, bodyB.shape)
        coll_data = bodyA.check_collision(bodyB)
        if coll_data['collision_count'].item():
            normal = coll_data['normal']
            v_rel_normal = coll_data['v_rel_normal']
            #only apply impulse if velocities are non-seperating
            #note: this part is only for speed, can be removed if desired
            if v_rel_normal.item() <= 0:
                e = min(bodyA.material.restitution, bodyB.material.restitution)
                normal_impulse_magn = - (1 + e) * v_rel_normal
                normal_impulse_magn /= bodyA.mass_data['inv_mass'] + bodyB.mass_data['inv_mass']
                normal_impulse = normal_impulse_magn * normal
                #apply to body
                bodyA.velocity -= normal_impulse * bodyA.mass_data['inv_mass']
                bodyB.velocity += normal_impulse * bodyB.mass_data['inv_mass']
            
                #apply frictional impules
                tangent = coll_data['tangent']
                v_rel_tangent = coll_data['v_rel_tangent']
                friction_impulse_magn = - v_rel_tangent
                friction_impulse_magn /= bodyA.mass_data['inv_mass'] + bodyB.mass_data['inv_mass']

                #note: could also take averae of the two mu's (actually that might be more interprettable)
                mu_static = np.sqrt(bodyA.material.mu_static ** 2 + bodyB.material.mu_static ** 2)            
                mu_dynamic = np.sqrt(bodyA.material.mu_dynamic ** 2 + bodyB.material.mu_dynamic ** 2)            

                coloumb_condition = torch.abs(friction_impulse_magn) < torch.abs(mu_static * normal_impulse_magn) 
                friction_impulse = torch.where(
                    coloumb_condition, 
                    friction_impulse_magn * tangent, 
                    -1.0 * mu_dynamic * normal_impulse_magn * tangent)
                print(coloumb_condition, friction_impulse)

                bodyA.velocity -= friction_impulse * bodyA.mass_data['inv_mass']
                bodyB.velocity += friction_impulse * bodyB.mass_data['inv_mass']


            # #apply positional correction
            self.positional_correction(bodyA, bodyB, coll_data)
        
        bodyA.update_pose(0.01) 
        bodyB.update_pose(0.01) 
        print('velocities', bodyA.velocity, bodyB.velocity)
    
    def positional_correction(self, bodyA, bodyB, coll_data):
        percent = 0.2 #20-80%
        slop = 0.01 #0.01 - 0.1
        penetration = coll_data['penetration']
        normal = coll_data['normal']
        total_inv_mass = bodyA.mass_data['inv_mass'] + bodyB.mass_data['inv_mass']
        correction = (max(penetration - slop, 0.) / total_inv_mass) * percent * normal
        bodyA.update_position_delta(-bodyA.mass_data['inv_mass'] * correction)
        bodyB.update_position_delta(bodyB.mass_data['inv_mass'] * correction)
        
    def init_viewer(self):
        if not self.vis_initialized:
            # from panda3d_viewer import Viewer, ViewerConfig
        
            # self.viewer_config = ViewerConfig()
            # self.viewer_config.set_window_size(320, 240)
            # self.viewer_config.enable_antialiasing(True, multisamples=4)
            # self.viewer = Viewer(window_type='onscreen', window_title='Sim', config=self.viewer_config)
            # self.viewer_initialized = True
            self.vis = meshcat.Visualizer() if self.vis is None else self.vis
            self.vis.open()
            self.vis_initialized = True



    def visualize(self):
        if self.vis_initialized:
            for k, v in self.bodies.items():
                position = v.pose.translation().cpu().numpy()
                radius = v.shape.radius.item()
                self.vis["world"][k].set_object(g.Sphere(radius))
                self.vis["world"][k].set_transform(tf.translation_matrix(position))

            # self.viewer.append_group('root')
            # # self.viewer.append_box('root', 'box_node', size=(1, 1, 1))
            # for k, v in self.spheres.items():
            #     self.viewer.append_sphere('root', k, radius=v.radius.item())
            # # self.viewer.set_material('root', 'box_node', color_rgba=(0.7, 0.1, 0.1, 1))
            #     self.viewer.set_material('root', k, color_rgba=(0.1, 0.7, 0.1, 1))

            # self.viewer.move_nodes('root', {
            #     'box_node': ((0, 0, 0.5), (1, 0, 0, 0)),
            #     'sphere_node': ((0, 0, 1.5), (1, 0, 0, 0))})

            # self.viewer.reset_camera(pos=(4, 4, 2), look_at=(0, 0, 1))



if __name__ == "__main__":
    import time
    sim = Sim()
    sim.init_viewer()
    while True:
        try:    
            sim.visualize()
            time.sleep(0.1)
            sim.step()
        except KeyboardInterrupt:
            exit()