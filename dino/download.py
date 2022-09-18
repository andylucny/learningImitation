import requests
import os

def download_image():
    if os.path.exists("probing_vits_9_0.png"):
        return
    print("downloading testing image")
    url = "https://keras.io/img/examples/vision/probing_vits/probing_vits_9_0.png"
    response = requests.get(url)
    open("probing_vits_9_0.png","wb").write(response.content)
    print("testing image downloaded")

def download_model():
    if os.path.exists("dino_deits8.onnx"):
        return
    print("downloading model")
    url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deits8.onnx"
    response = requests.get(url)
    open("dino_deits8.onnx","wb").write(response.content)
    print("model downloaded")


if __name__ == "__main__":
    download_image()
    download_model()
    print("done")
