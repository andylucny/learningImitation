import cv2
from agentspace import Agent,Space
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubGlobalCamera


class CameraAgent(Agent):

    def init(self):
        camera = iCubGlobalCamera()
        while True:
            # Grab a frame
            img = camera.grab()
            # sample it onto blackboard
            Space.write("camera",img)
 
    def senseSelectAct(self):
        pass

