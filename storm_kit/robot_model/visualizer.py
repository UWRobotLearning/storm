
#A wrapper around pinocchio Meshcat visualizer
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import sys 

class Visuailizer():
    def __init__(self):
        pass

    def initialize(self, robot_model):
        self.urdf = robot_model.urdf
        self.mesh_dir = robot_model.mesh_dir
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            self.urdf, self.mesh_dir, pin.JointModelFreeFlyer()
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
    
    def step(self, robot_model):
        pass




if __name__ == "__main__":
    from storm_kit.robot_model.robot_model import RobotModel
    robot_model = RobotModel("../../content/config/urdf/franka_description/franka_panda_no_gripper.urdf")
    # (self, urdf_path:str, mesh_dir:str="", name:str="", device:torch.device=torch.device('cpu')):
