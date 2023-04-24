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
        self.attach_trigger("features")
        self.acted = None
        self.last = 0.0
        self.maxk = 100 #200
  
    def generate(self):
        xx = np.random.uniform(-1,1)
        yy = np.random.uniform(-1,1)
        #xx = np.sign(xx)*(np.abs(xx)**2)
        #yy = np.sign(yy)*(np.abs(yy)**2)
        print("xx,yy",xx,yy)
        return (xx,yy)
        
    def add(self,query,proprio):
        contribute = True
        if len(self.keys) > 0:
            output, _ = Attention(query,self.keys,self.values,50.0)#np.sqrt(len(query)))
            diff = np.linalg.norm(np.array(output)-np.array(proprio))
            #print('diff',diff)
            eps = 0.2 #40/self.maxk #0.2...200, 0.1...400, 0.4..100, 0.8..50, eps = 40/n
            contribute = (diff > eps)
        n = 200
        pt = (int(proprio[0]*n+n),int(proprio[1]*n+n))
        if contribute:
            self.keys.append(query)
            self.values.append(proprio)
            k = len(self.keys)
            print('added',k-1)
            image = Space.read("camera",None)
            cv2.imwrite("pose"+str(k-1)+".png",image)
            cv2.circle(self.img,pt,3,(0,0,255),cv2.FILLED)
        else:
            cv2.circle(self.img,pt,3,(0,255,0),cv2.FILLED)
        cv2.imshow("postures",self.img)
        cv2.waitKey(1)
        return contribute
            
    def senseSelectAct(self):
        query = Space.read("features",None)
        if query is None:
            return
        proprio = Space.read("proprio",None)
        if proprio is None:
            return
        if self.acted is None:
            self.acted = proprio
        loss = np.linalg.norm(np.array(proprio)-np.array(self.acted))
        print('loss',loss)
        if loss < 0.01 or np.abs(loss - self.last) < 0.0005:
            act = self.generate()
            n = 200
            pt = (int(act[0]*n+n),int(act[1]*n+n))
            Space.write("act",act)
            self.acted = act
        else:
            if self.add(query,proprio):
                k = len(self.keys)
                if k % 10 == 0:
                    np.savetxt("keys.npy",np.array(self.keys))
                    np.savetxt("values.npy",np.array(self.values))
                    cv2.imwrite("points"+str(k)+".png",self.img)
                if k == self.maxk:
                    os._exit(0)
        self.last = loss