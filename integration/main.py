import os
os.environ['PATH'] += "iCumSim\\bin;"

from agentspace import Agent,Space
from CameraAgent import CameraAgent
from WhistleAgent import WhistleAgent
from PerceptionAgent import PerceptionAgent
from ActionAgent import ActionAgent
from ExpressionAgent import ExpressionAgent
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
sys.path.append('../dino')
from download import download_model
download_model()
decoder = 'vae-iCub-arms-decoder.pb'
if not os.path.exists(decoder):
    import shutil
    print('copying',decoder)
    shutil.copyfile('../vae/'+decoder, decoder)
    if not os.path.exists(decoder):
        print(decoder,'is missing')
        os._exit(0)

# get image from camera
CameraAgent()
ViewerAgent()
# get whistling
WhistleAgent()

PerceptionAgent() # dino encoder
ControlAgent() # attention
ActionAgent() # vae decoder

ExpressionAgent() # smile for invitation, neutral for imitation
