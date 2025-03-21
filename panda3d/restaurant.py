from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3, AmbientLight, DirectionalLight, Vec4
from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3

class UrbanEnvironment(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the urban environment model
        self.scene = self.loader.loadModel("models/urban_enviroment_gltf/scene.gltf")
        self.scene.reparentTo(self.render)  # Attach it to the scene

        # Apply scale and position transforms on the model.
        self.scene.setHpr(0, 90, 0)  # Rotate 90 degrees around the Y-axis
        self.scene.setScale(25, 20, 20)
        self.scene.setPos(0, 25, -2)


        # # Load the urban environment model
        # self.scene = self.loader.loadModel("models/uploads_files_4861689_Diner+(OBJ)/Diner.obj")
        # self.scene.reparentTo(self.render)  # Attach it to the scene

        # # Apply scale and position transforms on the model.
        # self.scene.setHpr(0, 90, 0)  # Rotate 90 degrees around the Y-axis
        # # self.scene.setScale(20, 20, 20)
        # self.scene.setPos(0, 25, -2)

        # Load and transform the panda actor.
        self.pandaActor = Actor("models/Car/Car.egg")
        self.pandaActor.setScale(0.1, 0.1, 0.1)
        self.pandaActor.reparentTo(self.render)
        self.pandaActor.setPos(0, 25, -2)



        


        # Add the spinCameraTask procedure to the task manager.
        # self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Set up lighting
        # self.setLighting()


    # Define a procedure to move the camera.
    # def spinCameraTask(self, task):
    #     angleDegrees = task.time * 6.0 # rotate 6 degrees every one second
    #     angleRadians = angleDegrees * (pi / 180.0) # convert to radian
    #     self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3) 
    #     # Y: horizontal, Z: vertical
    #     # Z is left fixed at 3 units above ground level
    #     self.camera.setHpr(angleDegrees, 0, 0)
    #     return Task.cont


app = UrbanEnvironment()
app.run()
