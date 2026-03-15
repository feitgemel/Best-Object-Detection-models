from ultralytics import YOLO
import torch

print("Check if Cuda avaiable :")
print(torch.cuda.is_available())
print(torch.__version__)
print("==================================")

model = YOLO('yolov8l.pt')

#  Convert the model to Tensor RT
# It will generate a "yolov8l.onnx" in my working folder 

model.export(format="engine", device='cuda')



