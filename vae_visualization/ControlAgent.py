import os
import numpy as np
from agentspace import Agent,Space
from time import sleep
import cv2

background = cv2.imread("bck.png")

def removeBackground(frame,newbck):
    mask = np.ones_like(frame)
    mask[background == frame] = 0
    ret = (mask * frame + (1-mask)*newbck).astype(np.uint8)
    return ret, mask

class ControlAgent(Agent):

    def init(self):
        resolution = (240,320)
        m = 40
        n = 200
        result = np.zeros((resolution[0]*(m+1),resolution[1]*(m+1),3),np.uint8)
        bck = np.zeros((resolution[0],resolution[1],3),np.uint8)
        bck[:,:] = (49,176,47)
        xs = np.linspace(0,2*n,m+1)
        for i, x in enumerate(xs):
            for j, y in enumerate(xs):
                xx = (x - n) / n
                yy = (y - n) / n
                act = (xx,yy)
                Space.write("act",act)
                sleep(2)
                frame = Space.read("view",None)
                if frame is not None:
                    cv2.imwrite("frame%03d%03d.png" % (x, y),frame)
                    image, _ = removeBackground(frame,bck)
                    cv2.imwrite("pose_%03d%03d.png" % (x, y),image)
                    result[j*resolution[0]:(j+1)*resolution[0],i*resolution[1]:(i+1)*resolution[1],:] = image
        
        print('done')
        cv2.imwrite("result.png",result)
        os._exit(0)

    def senseSelectAct(self):
        pass
        
