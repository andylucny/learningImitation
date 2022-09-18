import cv2
import numpy as np
from time import sleep
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubRightArm, iCubLeftArm

iCubApplicationName('/app/gener')

right_arm = iCubRightArm()
left_arm = iCubLeftArm()

net = cv2.dnn.readNet('vae-iCub-arms-decoder.pb')

active = False
points = []
def mouseHandler(event, x, y, flags, param):
    global refPt0, refPt1, selPt0, selPt1, active, points
    if event == cv2.EVENT_RBUTTONDOWN:
        active = True
    elif event == cv2.EVENT_RBUTTONUP:
        active = False
        points = []
    if active:
        points.append((x,y))
    elif event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        pass
        
cv2.namedWindow("select point")
cv2.setMouseCallback("select point", mouseHandler)

n = 200
img = np.ones([2*n,2*n,3],np.uint8)*255
cv2.line(img,(0,n),(2*n,n),(160,160,160),1)
cv2.line(img,(n,0),(n,2*n),(160,160,160),1)

nice_points=[(220,138),(202,197),(250,201),(131,200),(188,194),(141,261),(214,280),(107,269)]
k=0

while True:
    if len(points) > 0:
        x, y = points.pop()
        if not active:
            print(x,y)
            cv2.circle(img,(x,y),3,(0,0,255),cv2.FILLED)

        xx = (x - n) / n
        yy = (y - n) / n
        blobs = np.array([[xx,yy]])
        net.setInput(blobs)
        out = net.forward()
        joints = 180*out[0]
        #print(joints)
        right_joints = tuple(joints[:5].astype(np.double))
        left_joints = tuple(joints[5:].astype(np.double))
        right_arm.set(right_joints)
        left_arm.set(left_joints)
        
    cv2.imshow("select point",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
    elif key == ord('k'):
        print(k)
        points.append(nice_points[k])
        k += 1
        if k == len(nice_points):
            k = 0
        
cv2.imwrite('points.png',img)
cv2.destroyAllWindows()
 