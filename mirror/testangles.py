import cv2
import numpy as np
from time import sleep
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubRightArm, iCubLeftArm

iCubApplicationName('/app/angles')

simulator = "icubSim2"
right_arm = iCubRightArm(simulator)
left_arm = iCubLeftArm(simulator)

right_angles = right_arm.get()
left_angles = left_arm.get()

with np.printoptions(precision=0,suppress=True):
    print(np.array(right_angles))
    print(np.array(left_angles))

joints0 = np.array([-14.,  32.,  39.,  72.,  -0., -27., 123., -33.,  71.,   0.])
joints7 = np.array([-54.,  17.,  21.,  42.,  -0., -59., -11.,  45.,  43.,   0.])

def set(joints):
    global right_joints, left_joints
    right_joints = tuple(joints[:5].astype(np.double))
    left_joints = tuple(joints[5:].astype(np.double))
    right_arm.set(right_joints)
    left_arm.set(left_joints)

#set(joints0)
#sleep(3)
#set(joints7)

joints77 = np.array([-54.,  17.,  21.,  42.,  -0., -54.,  17.,  21.,  42.,  -0.])
set(joints77)

