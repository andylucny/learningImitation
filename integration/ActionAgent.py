from agentspace import Agent,Space
import numpy as np
import cv2
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubRightArm, iCubLeftArm

from time import sleep

class ActionAgent(Agent):

    def init(self):
        iCubApplicationName('/app/imitation')
        self.right_arm = iCubRightArm()
        self.left_arm = iCubLeftArm()
        self.net = cv2.dnn.readNet('vae-iCub-arms-decoder.pb')
        self.attach_trigger("act")
 
    def senseSelectAct(self):
        act = Space.read("act",None)
        if act is None:
            return

        blobs = np.array([act])
        self.net.setInput(blobs)
        out = self.net.forward()
        joints = 180*out[0]
        
        right_joints = tuple(joints[:5].astype(np.double))
        left_joints = tuple(joints[5:].astype(np.double))
        self.right_arm.set(right_joints)
        self.left_arm.set(left_joints)
        
        #Space.write("act",None,validity=2,priority=2) # do not allow the next actions for 2s
        sleep(0.5)
