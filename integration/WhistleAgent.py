import pyaudio
import numpy as np
from agentspace import Agent,Space

def H(x): # entropy
    return -np.sum(x*np.log(x))

class WhistleAgent(Agent):

    def init(self):
        rate = 44100 
        frames_per_buffer = 8*1024 
        channels = 1 
        format = pyaudio.paFloat32 
        audio = pyaudio.PyAudio() 
        stream = audio.open(format=format,channels=channels, rate=rate, input=True, frames_per_buffer=frames_per_buffer) 
        stream.start_stream()
        last = 0
        accum = 0
        while True: 
            audiobuffer = stream.read(frames_per_buffer) 
            signal = np.frombuffer(audiobuffer, dtype=np.float32) 
            s = signal * np.hamming(len(signal))
            f = np.fft.fft(s) 
            amp = np.abs(f)
            amp = amp[1:len(amp)//2] 
            pat = np.ones((len(amp)),amp.dtype) / len(amp)
            h = H(amp+pat) - H(amp)/2 - H(pat)/2
            silence = (h < 0)
            freq = np.argmax(amp)
            value = amp[freq]
            cnt = np.sum((amp > value/2).astype(int))
            #print(h,int(freq),int(value),cnt)
            whistling = silence and cnt <= 5 and freq > 200
            kind = int(whistling)
            accum += kind
            if last == 0 and kind == 0:
                if accum > 0:
                    if accum >= 5: 
                        print("long whistle")
                        Space.write("whistle", 2, validity=1)
                    else:
                        print("short whistle")
                        Space.write("whistle", 1, validity=1)
                accum = 0
            last = kind
            
    def senseSelectAct(self):
        pass
