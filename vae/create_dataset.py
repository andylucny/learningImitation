import numpy as np

fname = '../kinematics/rightarm_wrist.txt'
with open(fname,'r') as f:
    lines = f.readlines()
    
data = [line[:-1].split(',')[:-1] for line in lines]
data = np.array(data,np.float32)

n = 10000
samples = np.zeros([n,10+6],np.float32)
for i in range(n):
    if np.random.randint(0,2) == 0:
        j = np.random.randint(0,len(data))
        samples[i,:5] = data[j,6:11]
        samples[i,5:10] = data[j,6:11]
        samples[i,10:13] = data[j,0:3]
        samples[i,13:] = data[j,0:3]
    else:
        j = np.random.randint(0,len(data))
        k = np.random.randint(0,len(data))
        samples[i,:5] = data[j,6:11]
        samples[i,5:10] = data[k,6:11]
        samples[i,10:13] = data[j,0:3]
        samples[i,13:] = data[k,0:3]

np.savetxt('dataset.npy',samples,fmt='%1.3f')
