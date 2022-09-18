import sys
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubHead, iCubTorso, iCubRightArm, iCubLeftArm, iCubRightLeg, iCubLeftLeg

iCubApplicationName('/renewer')

head = iCubHead()
head.set((0,0,0,0,0,0))

torso = iCubTorso()
torso.set((0,0,0))

right_arm = iCubRightArm()
right_arm.set((0,80,0,50,0,0,0,59,20,20,20,10,10,10,10,10))

left_arm = iCubLeftArm()
left_arm.set((0,80,0,50,0,0,0,59,20,20,20,10,10,10,10,10))

left_leg = iCubRightLeg()
left_leg.set((0,0,0,0,0,0))

left_leg = iCubLeftLeg()
left_leg.set((0,0,0,0,0,0))
