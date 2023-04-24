from agentspace import Agent,Space
import numpy as np
import cv2
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubRightArm, iCubLeftArm, checkArmConstraints

class ProprioceptionAgent(Agent):

    def init(self):
        iCubApplicationName('/app/imitation')
        self.right_arm = iCubRightArm()
        self.left_arm = iCubLeftArm()
        self.net = cv2.dnn.readNet('vae-iCub-arms-simplified-encoder.pb')
        self.attach_trigger("camera")
 
    def senseSelectAct(self):
        right_joints = self.right_arm.get()
        left_joints = self.left_arm.get()
        joints = right_joints[:5] + left_joints[:5]
        blobs = np.array([joints])/180.0
        self.net.setInput(blobs)
        out = self.net.forward()
        features = tuple(out[0])
        Space.write("proprio",features)
