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

last_c = None    
def logging(c):
    global last_c
    eps = 0.01
    change = 2*eps if last_c is None else np.linalg.norm(np.array(last_c)-np.array(c))/len(c) 
    #if change > 0:
    #   print(change)
    if change > eps:
        with np.printoptions(precision=2,suppress=True):
            print(np.array(c))
        last_c = c

last_a = None    
def logging2(a):
    global last_a
    eps = 0.01
    change = 2*eps if last_a is None else np.linalg.norm(np.array(last_a)-np.array(a))/len(a) 
    if change > eps:
        print('change','{:.6f}'.format(change),a)
        with np.printoptions(precision=2,suppress=True):
            with open('actions.txt', 'a') as f:
                np.savetxt(f,np.array(a))
        last_a = a

class ControlAgent(Agent):

    def init(self):
        self.keys = np.loadtxt("keys.npy")
        self.values = np.loadtxt("values.npy")
        self.attach_trigger("features")
  
    def senseSelectAct(self):
        query = Space.read("features",None)
        if query is None:
            return
        act, coefs = Attention(query,self.keys,self.values,len(query)**0.5)#1.0/len(query)) #1.0/(len(query)**2)) #len(query)**0.5) # 1.0/len(query))#0.2 #len(query)**0.5        
        Space.write("act",act)
        logging(coefs)
        logging2(act)
