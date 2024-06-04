import torch 
import detectron2

import numpy as np 
import cv2 

from detectron2 import model_zoo 
from detectron2.engine import DefaultPredictor 
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer 
from detectron2.data import MetadataCatalog

imagePath = "Best-Object-Detection-models/Detectron2/Simple-Object_detection/pexels-brett-sayles-1115171.jpg"
img = cv2.imread(imagePath)

# reduce the image size
scale_precent = 30 
width = int(img.shape[1] * scale_precent / 100)
height = int(img.shape[0] * scale_precent / 100)

dim = (width, height)

myNewImage = cv2.resize(img, dim , interpolation = cv2.INTER_AREA)




# create a new config file
cfg_keypoint = get_cfg()


# find the models for object detection:
# link for the Detectron2 models : https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

cfg_keypoint.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg_keypoint.MODEL.ROI_HEADS.SCORE_TRESH_TEST = 0.5  # higher means better propability

# get the Weights
cfg_keypoint.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg_keypoint.MODEL.DEVICE = "cpu" # if you run on Linux and have Cuda , use can delete this line


predictor = DefaultPredictor(cfg_keypoint)
outputs = predictor(myNewImage)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(myNewImage[:,:,::-1], MetadataCatalog.get(cfg_keypoint.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img = out.get_image()[:, :, ::-1]

cv2.imshow("img", myNewImage)
cv2.imshow("predict", img)

cv2.waitKey(0)
cv2.destroyAllWindows()