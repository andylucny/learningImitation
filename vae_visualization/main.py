import os
os.environ['PATH'] += "iCumSim\\bin;"

from agentspace import Agent,Space
from RecorderAgent import RecorderAgent
from ActionAgent import ActionAgent
from ControlAgent import ControlAgent
from ViewerAgent import ViewerAgent
import signal
import time

# exit on ctrl-c
def signal_handler(signal, frame):
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# check resources
import sys
decoder = 'vae-iCub-arms-decoder.pb'
if not os.path.exists(decoder):
    import shutil
    print('copying',decoder)
    shutil.copyfile('../vae/'+decoder, decoder)
    if not os.path.exists(decoder):
        print(decoder,'is missing')
        os._exit(0)
from bgremover import download
download()

# get image from the global camera
RecorderAgent()
ViewerAgent()

ControlAgent() # attention
ActionAgent() # vae decoder

