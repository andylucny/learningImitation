import cv2
import numpy as np
import time 
import os
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubRightArm, iCubLeftArm, iCubGlobalCamera, checkArmConstraints, Kinematics
from CameraAgent import CameraAgent
from agentspace import Agent,Space

def quit():
    os._exit(0)
    
iCubApplicationName('/app/gener')

right_arm = iCubRightArm("icubSim2")
left_arm = iCubLeftArm("icubSim2")
right_arm1 = iCubRightArm("icubSim")
left_arm1 = iCubLeftArm("icubSim")

CameraAgent("icubSim","camera")
CameraAgent("icubSim2","camera2")

net = cv2.dnn.readNet('vae-iCub-arms-decoder.pb')

auto = True

point = (0,0)
lbuttonpressed = False
def mouseHandler(event, x, y, flags, param):
    global refPt0, refPt1, selPt0, selPt1, point, lbuttonpressed, auto 
    if event == cv2.EVENT_LBUTTONDOWN:
        lbuttonpressed = True
    elif event == cv2.EVENT_LBUTTONUP:
        lbuttonpressed = False
    if lbuttonpressed:
        if not auto:
            print("x,y",x,y)
            point = (x,y)
        
cv2.namedWindow("select point")
cv2.setMouseCallback("select point", mouseHandler)

n = 100 # 200
img = np.ones([2*n,2*n,3],np.uint8)*255
cv2.line(img,(0,n),(2*n,n),(160,160,160),1)
cv2.line(img,(n,0),(n,2*n),(160,160,160),1)

nice_index=-1
nice_points=[(220,138),(202,197),(250,201),(131,200),(188,194),(141,261),(214,280),(139,227)]#(107,269)]
k=0

accs = []
t = time.time()+3.0
last_point = point
repeat = False
while True:
    x, y = point

    if repeat or last_point[0] != point[0] or last_point[1] != point[1]:
        repeat = False
        last_point = point
        xx = (x - n) / n
        yy = (y - n) / n
        blobs = np.array([[xx,yy]])
        net.setInput(blobs)
        out = net.forward()
        joints = 180*out[0]
        with np.printoptions(precision=0,suppress=True):
            print("{:.4f}".format(xx),"{:.4f}".format(yy),np.array(joints))
        right_joints = tuple(joints[:5].astype(np.double))
        left_joints = tuple(joints[5:].astype(np.double))
        right_joints = checkArmConstraints(right_joints)
        left_joints = checkArmConstraints(left_joints)
        right_arm.set(right_joints)
        left_arm.set(left_joints)
        
    img2 = np.copy(img)
    cv2.circle(img2,point,3,(0,0,255),cv2.FILLED)
    cv2.imshow("select point",img2)
    key = cv2.waitKey(10)
    if auto and time.time() > t:
        if nice_index != -1:
            im2 = Space.read("camera2",None)
            im = Space.read("camera",None)
            angles2 = right_arm.get()[:5]+left_arm.get()[:5]
            angles = right_arm1.get()[:5]+left_arm1.get()[:5]
            ranges = Kinematics.rangesArm[:5]+Kinematics.rangesArm[:5]
            szs = [np.abs(range[0]-range[1]) for range in ranges]
            acc = np.average((np.array(szs) - np.abs(np.array(angles2)-np.array(angles)))/np.array(szs))
            acc = int(acc*100)
            accs.append(acc)
            print(nice_index,"acc",acc)
            cv2.putText(im, str(acc)+"%", (8,22), 0, 0.9, (0,0,255), 1, cv2.LINE_AA)
            hh, ww = im2.shape[:2]
            hhh = hh // 5
            www = ww // 5
            im2[8:8+hhh,8:8+www] = cv2.resize(img2,(www,hhh))
            cv2.rectangle(im2, (0,0,ww-1,hh-1), (0,0,0), 1)
            hh, ww = im.shape[:2]
            cv2.rectangle(im, (0,0,ww-1,hh-1), (0,0,0), 1)
            cv2.imwrite("pattern"+str(nice_index)+".png",im2)
            cv2.imwrite("image"+str(nice_index)+".png",im)
        key = ord('n')
        t = time.time()+3.0
    if key == 27:
        break
    elif key == ord('n'):
        nice_index += 1
        if nice_index == len(nice_points):
            if auto:
                print("result acc",np.average(accs))
                break
            nice_index = 0
        point = nice_points[nice_index]
        point = (point[0]//2, point[1]//2)
        print("nice point",nice_index,"x,y:",point[0],point[1])
    elif key == ord('p'):
        nice_index -= 1
        if nice_index < 0: 
            nice_index = len(nice_points)-1
        point = nice_points[nice_index]
        point = (point[0]//2, point[1]//2)
        print("nice point",nice_index,"x,y:",point[0],point[1])
    elif key == ord('r'):
        point = nice_points[nice_index]
        point = (point[0]//2, point[1]//2)
        print("nice point",nice_index,"x,y:",point[0],point[1])
        repeat = True

cv2.destroyAllWindows()
