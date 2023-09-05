from typing import Dict
from storm_kit.geom.shapes import Sphere
import torch

#
def sphere_sphere_collision(sphere1: Sphere, sphere2: Sphere) -> Dict[str, torch.Tensor]:
    # device=sphere1.radius.device
    # batch_size=sphere2.radius.shape[0]

    #find contact frame
    normal = sphere2.pose.translation() - sphere1.pose.translation()
    
    r = sphere1.radius + sphere2.radius
    dist = torch.linalg.vector_norm(normal, ord=2, dim=-1, keepdims=True)
    
    collision_count = torch.where(dist < r, 1.0, 0.0)
    penetration = torch.where(dist == 0.0, sphere1.radius, r - dist)
    I = torch.eye(normal.shape[-2], normal.shape[-1], device=normal.device)
    # I = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(batch_size,1)    
    collision_normal = torch.where(dist==0.0, I, normal/dist)
    
    #TODO?: Add contact point calculation

    return {
        'normal': collision_normal,
        'penetration': penetration,
        'collision_count': collision_count,
        'dist': dist
    }


if __name__ == "__main__":
    device=torch.device('cuda:0')
    
    #two spheres at same position
    print('Two spheres at same position')
    A = Sphere(
        position=torch.tensor([[0.0, 0.0, 0.0]], device=device), 
        radius=torch.tensor([[1.0]], device=device))

    B = Sphere(
        position=torch.tensor([[0.0, 0.0, 0.0]], device=device), 
        radius=torch.tensor([[1.0]], device=device))
    
    print(sphere_sphere_collision(A, B))

    #two non-collidiing spheres 
    print('Two non-colliding spheres')
    A = Sphere(
        position=torch.tensor([[0.0, 0.0, 0.0]], device=device), 
        radius=torch.tensor([[1.0]], device=device))

    B = Sphere(
        position=torch.tensor([[2.0, 2.0, 3.0]], device=device), 
        radius=torch.tensor([[1.0]], device=device))
    
    print(sphere_sphere_collision(A, B))

    #two collidiing spheres 
    print('Two colliding spheres')
    A = Sphere(
        position=torch.tensor([[0.0, 0.0, 0.0]], device=device), 
        radius=torch.tensor([[1.0]], device=device))

    B = Sphere(
        position=torch.tensor([[1.5, 0.0, 0.0]], device=device), 
        radius=torch.tensor([[1.0]], device=device))
    
    print(sphere_sphere_collision(A, B))

    #one sphere with a batch of random spheres
    print('One sphere with a batch of random spheres') 
    A = Sphere(
        position=torch.tensor([[0.0, 0.0, 0.0]], device=device), 
        radius=torch.tensor([[1.0]], device=device))

    B = Sphere(
        position=torch.randn(5,3, device=device), 
        radius=torch.randn(5,1, device=device))
    
    print(sphere_sphere_collision(A, B))

