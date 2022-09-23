import numpy as np
from agentspace import Agent,Space
from time import sleep
from keras.backend import softmax
import cv2

def Attention(query,keys,values,d):
    keys_matrix = np.array(keys,np.float32)
    values_matrix = np.array(values,np.float32)
    c = softmax(query.dot(keys_matrix.T)/d).numpy()
    output = c.dot(values_matrix)
    return output

class ControlAgent(Agent):

    def init(self):
        self.keys = [] #np.zeros([0,384],np.float32) #np.loadtxt("keys.npy")
        self.values = [] #np.zeros([0,2],np.float32) #np.loadtxt("values.npy")
        self.mode = 0
        self.acted = None
        self.k = -1
        self.attach_trigger("whistle")
        self.attach_trigger("features")
        
    def generate(self,predefined=False):
        if predefined:
            nice_points=[(220,138),(202,197),(250,201),(131,200),(188,194),(141,261),(214,280),(107,269)]
            self.k += 1
            if self.k == len(nice_points):
                self.k = 0
            x, y = nice_points[self.k]
            n = 200
            xx = (x - n) / n
            yy = (y - n) / n
        else:
            xx = np.random.uniform(-1,1)
            yy = np.random.uniform(-1,1)
        return (xx,yy)
            
    def senseSelectAct(self):
        whistling = Space.read("whistle",0)
        if whistling > 0:
            Space.write("whistle",0,validity=1,priority=2)
            if whistling == 2:
                self.mode = 1-self.mode
                if self.mode == 0:
                    self.keys = []
                    self.values = []
                    self.k = -1

        query = Space.read("features",None)
        if query is None:
            return
            
        Space.write("mode",self.mode)
        if self.mode == 0: # invitation

            if self.acted is None:
                print("inviting to imitation")
                act = self.generate(True)
                Space.write("act",act)
                self.acted = act
           
            else:
                if whistling == 1:
                    print("mapping")
                    self.keys.append(query)
                    self.values.append(self.acted)
                    #np.savetxt("keys.npy",np.array(self.keys))
                    #np.savetxt("values.npy",np.array(self.values))
                    self.acted = None
                    Space.write("confirm",True,validity=0.15)
        
        else: # self.mode == 1 # imitation

            act = Attention(query,self.keys,self.values,len(query)**0.2)
            Space.write("act",act)
