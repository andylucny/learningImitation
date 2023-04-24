import pyaudio
import numpy as np
from agentspace import Agent,Space
from time import sleep, time
import cv2

class ViewerAgent(Agent):

    def __init__(self, title="input image"):
        self.title = title
        super().__init__()

    def init(self):
        self.attach_trigger("camera")
            
    def senseSelectAct(self):
        image = Space.read("camera",None)
        if image is not None:
            cv2.imshow(self.title,image)
            if cv2.waitKey(1) == ord('s'):
                cv2.imwrite('global'+str(time())+'.png',image)
