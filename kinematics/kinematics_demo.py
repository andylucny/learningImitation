from time import sleep
import numpy as np
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubTorso, iCubRightArm, Kinematics

iCubApplicationName('/app/demo')
torso = iCubTorso()
right_arm = iCubRightArm()

def d(p,q):
    return np.linalg.norm(np.array(p)-np.array(q))

def getPos():
    joints = right_arm.get()
    torso_joints = torso.get()
    positions = Kinematics.directRightArm(torso_joints+joints)
    return positions[Kinematics.wrist]

def setPos(goal):
    joints, achieved, distance, iterations = Kinematics.inverseRightArm(goal,Kinematics.shoulder,Kinematics.wrist)
    print('---', achieved, distance, iterations, goal)
    if achieved:
        torso.set(joints[:3])
        right_arm.set(joints[3:])
        for u in range(24):
            sleep(0.5)
            actual_distance = d(getPos(),goal)
            print(actual_distance)
            if actual_distance <= distance + 0.1: 
                break

print('move with the right arm')
last = getPos()
t = 0
while t < 5:
    sleep(1)
    pos = getPos()
    if d(last,pos) > 5:
        setPos(last)
        setPos(pos)
        sleep(1)
        last = pos
        t += 1
        print('move with the right arm again')

print('thank you, play with p = getPos() and setPos(p) from the command line')
