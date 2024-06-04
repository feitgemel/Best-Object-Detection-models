from detectron2.engine import DefaultPredictor 
import os 
import pickle 

cfg_save_path = "My-Train-OD-Detectron2/OD_cfg.pickle"
with open (cfg_save_path , 'rb') as f :
    cfg = pickle.load(f) # load the configuration file 


output_dir = r"My-Train-OD-Detectron2"
cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # prediction threshold 
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

CLASSES = ['Apple', 'Strawberry' , 'Orange', 'Grapes', 'Banana', 'Lemon']

# path to the test image 
#image_path = "Train-custom-Object-Detection-model/Fruits_for_detectron2/Test/pexels-pixabay-70746.jpg" 
image_path = "Train-custom-Object-Detection-model/Fruits_for_detectron2/Test/apples-vs-bananas.jpg" 


import cv2 
import numpy as np 
from detectron2.utils.visualizer import Visualizer 

im = cv2.imread(image_path)
outputs = predictor(im)

print("=========")
print(outputs)
print("=========")


pred_classes = outputs['instances'].pred_classes.cpu()
print("pred_classes : ")
print(pred_classes)

pred_classes = pred_classes.numpy()
print("pred_classes as Numpy : ")
print(pred_classes)

flag = np.size(pred_classes)
print("Flag : ")
print(flag)

if flag > 0 :

    pred_classes = pred_classes[0] # get the first one 
    print("pred_classes : ")
    print(pred_classes)

    print(CLASSES[pred_classes])

    img_RGB = cv2.cvtColor(im , cv2.COLOR_BGR2RGB)
    v = Visualizer(img_RGB , metadata={} , scale=0.4) # ini the Visualizer 
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img_bgr = cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR)

    cv2.imshow("v", img_bgr)
    cv2.waitKey(0)
else:
    print("Pred class is empty")

cv2.destroyAllWindows()































