
#A wrapper around pinocchio Meshcat visualizer
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import sys 

class Visualizer():
    def __init__(self):
        self.initialized = False

    def init_viewer(self, robot_model):
        self.urdf = robot_model.urdf
        self.mesh_dir = robot_model.mesh_dir
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            self.urdf, self.mesh_dir,
        )
        self.viz = MeshcatVisualizer(model, collision_model, visual_model)

        try:
            self.viz.initViewer(open=True)
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)
        
        # Load the robot in the viewer.
        self.viz.loadViewerModel()
        q0 = pin.neutral(model)
        print(q0)
        q = pin.randomConfiguration(model)
        print('q: %s' % q.T)
    
    def step(self, robot_model):
        if not self.initialized:
            self.init_viewer(robot_model)
        q_pos = robot_model.q_pos.cpu().numpy()
        self.viz.display(q_pos[0])
        self.viz.displayVisuals(True)

        

if __name__ == "__main__":
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    import torch
    from storm_kit.differentiable_robot_model.robot_model import DifferentiableRobotModel
    
    with initialize(config_path="../../content/configs/gym", job_name="test_app"):
        cfg = compose(config_name="config", overrides=["task=FrankaReacher"])
    
    robot_model = DifferentiableRobotModel(cfg.robot)
    viz = Visualizer()
    q0 = torch.tensor([0.0, -0.7853, 0.0, -2.3561, 0.0, 1.5707, 0.7853]).unsqueeze(0)
    # q0 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unsqueeze(0)

    robot_model.compute_fk_and_jacobian(q0, torch.zeros_like(q0), link_name="ee_link")
    viz.step(robot_model)
    input('....')