import pyaudio
import numpy as np
from agentspace import Agent,Space
from time import sleep
import cv2

class ViewerAgent(Agent):

    def init(self):
        self.attach_trigger("view")
            
    def senseSelectAct(self):
        image = Space.read("view",None)
        if image is not None:
            cv2.imshow("view",image)
            cv2.waitKey(1)
