import os
os.environ['PATH'] += "iCumSim\\bin;"

from agentspace import Agent,Space
from CameraAgent import CameraAgent
from PerceptionAgent import PerceptionAgent
from ActionAgent import ActionAgent
from ControlAgent1 import ControlAgent
from ViewerAgent import ViewerAgent
import signal
import time

# exit on ctrl-c
def signal_handler(signal, frame):
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

def quit():
    os._exit(0)
    
# check resources
import sys
sys.path.append('../dino')
from download import download_model
download_model()
def copy_model(srcdir,filename):
    if not os.path.exists(filename):
        import shutil
        print('copying',filename)
        shutil.copyfile(srcdir+filename,filename)
        if not os.path.exists(filename):
            print(filename,'is missing')
            os._exit(0)
copy_model('../vae/','vae-iCub-arms-decoder.pb')
copy_model('../vae/','vae-iCub-arms-simplified-encoder.pb')

# get image from camera
CameraAgent()
ViewerAgent("mirror image")

time.sleep(1)
PerceptionAgent() # dino encoder
ActionAgent() # vae decoder

time.sleep(6)
ControlAgent() # attention


