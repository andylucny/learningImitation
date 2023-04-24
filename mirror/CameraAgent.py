import cv2
from agentspace import Agent,Space
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubGlobalCamera


class CameraAgent(Agent):

    def __init__(self, simulator="icubSim", name="camera"):
        self.simulator = simulator
        self.name = name
        super().__init__()

    def init(self, ):
        camera = iCubGlobalCamera(self.simulator)
        while True:
            # Grab a frame
            img = camera.grab()
            # sample it onto blackboard
            Space.write(self.name,img)
 
    def senseSelectAct(self):
        pass

