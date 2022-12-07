from agentspace import Agent,Space
import numpy as np
import cv2
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubGlobalCamera

from time import sleep

class RecorderAgent(Agent):

    def init(self):
        iCubApplicationName('/app/imitation')
        camera = iCubGlobalCamera()
        while True:
            frame = camera.grab()
            Space.write("view",frame,validity=0.5) 

    def senseSelectAct(self):
        pass
        
