import numpy as np
from agentspace import Agent,Space
import time
import os
from keras.backend import softmax
import cv2

nice_points=[(220,138),(202,197),(250,201),(131,200),(188,194),(141,261),(214,280),(139,227)]#,(107,269)]

class ControlAgent(Agent):

    def init(self):
        self.keys = [] 
        self.values = [] 
        n = 200
        self.img = np.ones([2*n,2*n,3],np.uint8)*255
        cv2.line(self.img,(0,n),(2*n,n),(160,160,160),1)
        cv2.line(self.img,(n,0),(n,2*n),(160,160,160),1)
        cv2.imshow("postures",self.img)
        cv2.waitKey(1)
        self.k = 0
        time.sleep(5)
        self.attach_timer(5)
  
    def generate(self,predefined=False):
        if predefined:
            print("nice point",self.k)
            x, y = nice_points[self.k]
            n = 200
            xx = (x - n) / n
            yy = (y - n) / n
        else:
            xx = np.random.uniform(-1,1)
            yy = np.random.uniform(-1,1)
        return (xx,yy)
            
    def senseSelectAct(self):
        act = self.generate(True)
        Space.write("act",act)
        self.acted = act
        time.sleep(3)
        image = Space.read("camera",None)
        query = Space.read("features",None)
        self.keys.append(query)
        self.values.append(self.acted)
        cv2.imwrite("pose"+str(self.k)+".png",image)
        n = 200
        pt = (int(self.acted[0]*n+n),int(self.acted[1]*n+n))
        cv2.circle(self.img,pt,3,(0,0,255),cv2.FILLED)
        cv2.imshow("postures",self.img)
        cv2.waitKey(1)

        self.k += 1
        if self.k == len(nice_points):
            np.savetxt("keys.npy",np.array(self.keys))
            np.savetxt("values.npy",np.array(self.values))
            time.sleep(3)
            os._exit(0)
