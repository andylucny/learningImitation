import numpy as np
from keras.backend import softmax

def Attention(query,keys,values,d):
    keys_matrix = np.array(keys,np.float32)
    values_matrix = np.array(values,np.float32)
    c = softmax(query.dot(keys_matrix.T)/d).numpy()
    output = c.dot(values_matrix)
    return output, c

keys = np.loadtxt("keys.npy")
values = np.loadtxt("values.npy")

d = np.sqrt(len(keys[0]))
for n in range(1,len(keys)+1):
    for i in range(len(keys)):
        with np.printoptions(precision=2,suppress=True):
            _, coefs = Attention(keys[i],keys[:n],values[:n],d)
            print(coefs)
    print("")