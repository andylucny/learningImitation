import numpy as np
from agentspace import Agent,Space
import time
import os
from keras.backend import softmax
import cv2

def Attention(query,keys,values,d):
    keys_matrix = np.array(keys,np.float32)
    values_matrix = np.array(values,np.float32)
    c = softmax(query.dot(keys_matrix.T)/d).numpy()
    output = c.dot(values_matrix)
    return output, c
    
class ControlAgent(Agent):

    def init(self):
        self.keys = [] 
        self.values = [] 
        n = 200
        self.img = np.ones([2*n,2*n,3],np.uint8)*255
        cv2.line(self.img,(0,n),(2*n,n),(160,160,160),1)
        cv2.line(self.img,(n,0),(n,2*n),(160,160,160),1)
        print('STARTING')
        for k in range(400):
            act = self.generate()
            pt = (int(act[0]*n+n),int(act[1]*n+n))
            Space.write("act",act)
            self.acted = act
            time.sleep(3)
            query = Space.read("features",None)
            add = True
            if len(self.keys) > 0:
                output, _ = Attention(query,self.keys,self.values,np.sqrt(len(query)))
                diff = np.linalg.norm(np.array(output)-np.array(self.acted))
                print('output',output)
                print('acted',self.acted)
                print('diff',diff)
                eps = 0.1
                add = (diff > eps)
            if add:
                self.keys.append(query)
                self.values.append(self.acted)
                print(k)
                image = Space.read("camera",None)
                cv2.imwrite("pose"+str(k)+".png",image)
                cv2.circle(self.img,pt,3,(0,0,255),cv2.FILLED)
            else:
                cv2.circle(self.img,pt,3,(0,255,0),cv2.FILLED)
            cv2.imshow("postures",self.img)
            cv2.waitKey(1)
        np.savetxt("keys.npy",np.array(self.keys))
        np.savetxt("values.npy",np.array(self.values))
        cv2.imwrite("points.png",self.img)
        os._exit(0)
  
    def generate(self):
        xx = np.random.uniform(-1,1)
        yy = np.random.uniform(-1,1)
        #xx = np.sign(xx)*(np.abs(xx)**2)
        #yy = np.sign(yy)*(np.abs(yy)**2)
        return (xx,yy)
            
    def senseSelectAct(self):
        pass
