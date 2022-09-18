from agentspace import Agent,Space
import sys
sys.path.append('../pyicubsim')
from pyicubsim import iCubApplicationName, iCubEmotion
from time import sleep

class ExpressionAgent(Agent):

    def init(self):
        iCubApplicationName('/app/imitation')
        self.emotion = iCubEmotion()
        self.emotion.set('hap')
        self.mode = 0
        self.confirmed = False
        self.attach_trigger("mode")
        self.attach_trigger("confirm")
 
    def senseSelectAct(self):
        confirm = Space.read("confirm",False)
        if confirm and not self.confirmed:
            self.confirmed = True
            print("confirming")
            self.emotion.set('sur')
            sleep(1)
            self.mode = -1
        if not confirm:
            self.confirmed = False
        mode = Space.read("mode",0)
        if mode != self.mode:
            self.emotion.set('neu' if mode > 0 else 'hap')
            sleep(0.15)
            self.mode = mode
            print("mode","invitation" if mode == 0 else "imitation")
