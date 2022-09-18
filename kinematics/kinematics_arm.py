from time import sleep
import numpy as np
import sys
sys.path.append('../pyicubsim')
from pyicubsim import Kinematics
import itertools

with open('rightarm_wrist.txt','w') as f:
#with open('rightarm_elbow.txt','w') as f:
    
    # wrist
    xs = np.linspace(-300,200,51)
    ys = np.linspace(-60,400,47)
    zs = np.linspace(-110,470,59) 
    
    # elbow
    #xs = np.linspace(-180,130,26) 
    #ys = np.linspace(-50,300,36)
    #zs = np.linspace(-100,430,54)

    for goal in itertools.product(xs,ys,zs):
        joints, achieved, eps, iterations = Kinematics.inverseRightArm(goal,Kinematics.shoulder,Kinematics.wrist) # hand
        #joints, achieved, eps, iterations = Kinematics.inverseRightArm(goal,Kinematics.shoulder,Kinematics.elbow) # elbow
        print(goal,'...',eps)
        if achieved:
            data = goal + joints + (iterations,)
            print("-->",data)
            f.write(str(data)[1:-1]+'\n')

# collect data
