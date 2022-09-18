import cv2
from agentspace import Agent,Space

class CameraAgent(Agent):

    def init(self):
        camera = cv2.VideoCapture(0)
        while True:
            # Grab a frame
            ret, img = camera.read()
            if not ret:
                self.stop()
                return
            
            # sample it onto blackboard
            Space.write("camera",img)
 
    def senseSelectAct(self):
        pass

