import cv2
import numpy as np
from time import sleep
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubRightArm, iCubLeftArm, iCubGlobalCamera

iCubApplicationName('/app/action')

right_arm = iCubRightArm("icubSim2")
left_arm = iCubLeftArm("icubSim2")

net = cv2.dnn.readNet('vae-iCub-arms-decoder.pb')

def action(act):
    xx, yy = act
    blobs = np.array([[xx,yy]])
    net.setInput(blobs)
    out = net.forward()
    joints = 180*out[0]
    #print(joints)
    right_joints = tuple(joints[:5].astype(np.double))
    left_joints = tuple(joints[5:].astype(np.double))
    right_arm.set(right_joints)
    left_arm.set(left_joints)

coefs1 = [0.,   0.,   0.,   0.,   0.02, 0.,   0.,   0.98]
act1 = [-0.45658866,  0.3372005 ]

coefs2 = [0.,   0.,   0.,   0.,   0.01, 0.,   0.,   0.99]
act2 = [-0.46194586,  0.3421575 ]

#action(act1)
#action(act2)
