from agentspace import Agent,Space
import numpy as np
import onnxruntime as ort
import cv2

class PerceptionAgent(Agent):

    def init(self):
        self.session = ort.InferenceSession("dino_deits8.onnx", providers=['CUDAExecutionProvider'])
        self.input_names = [input.name for input in self.session.get_inputs()] # ['x.1']
        self.output_names = [output.name for output in self.session.get_outputs()] # ['1158']
        self.attach_trigger("camera")
        self.last = np.zeros((384,),np.float32)
 
    def senseSelectAct(self):
        frame = Space.read("camera",None)
        if frame is None:
            return

        image_size = (224, 224)
        blob = cv2.dnn.blobFromImage(frame, 1.0/255, image_size, (0, 0, 0), swapRB=True, crop=True)
        blob[0][0] = (blob[0][0] - 0.485)/0.229
        blob[0][1] = (blob[0][1] - 0.456)/0.224
        blob[0][2] = (blob[0][2] - 0.406)/0.225
        
        data_input = { self.input_names[0] : blob }
        data_output = self.session.run(self.output_names, data_input)[0]
        features = data_output[0]
        #print(features.shape,np.max(np.abs(features-self.last)))
        
        self.last = features
        
        Space.write("features",features)
