import time
import numpy as np
import onnxruntime as ort
import cv2

session = ort.InferenceSession("dino_deits8.onnx", providers=['CUDAExecutionProvider'])

input_names = [input.name for input in session.get_inputs()] # ['x.1']
output_names = [output.name for output in session.get_outputs()] # ['1158']

path = "probing_vits_9_0.png"
frame = cv2.imread(path,cv2.IMREAD_COLOR)
image_size = (224, 224)
blob = cv2.dnn.blobFromImage(frame, 1.0/255, image_size, (0, 0, 0), swapRB=True, crop=False)
blob[0][0] = (blob[0][0] - 0.485)/0.229
blob[0][1] = (blob[0][1] - 0.456)/0.224
blob[0][2] = (blob[0][2] - 0.406)/0.225

for t in range(10):
    t0 = time.time()
    data_input = { input_names[0] : blob }
    data_output = session.run(output_names, data_input)[0]
    t1 = time.time()
    print(data_output.shape,' ... ',t1-t0) # 1x384 0.06

print('OK')
