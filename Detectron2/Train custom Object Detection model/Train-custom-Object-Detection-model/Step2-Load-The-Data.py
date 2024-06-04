import detectron2 
import numpy as np 
import cv2 

from detectron2.utils.visualizer import Visualizer 
from detectron2.data import MetadataCatalog , DatasetCatalog 

# Register to the train data and validation data 

from detectron2.data.datasets import register_coco_instances 

# Train dataset
register_coco_instances("my_dataset_train", {} , 
                        "Train-custom-Object-Detection-model/Fruits_for_detectron2/Train/labels_my-project-name_2023-12-04-07-26-09.json",
                        "Train-custom-Object-Detection-model/Fruits_for_detectron2/Train")

# Validation Dataset 
register_coco_instances("my_dataset_val", {} ,
                        "Train-custom-Object-Detection-model/Fruits_for_detectron2/Validate/labels_my-project-name_2023-12-04-07-39-25.json",
                        "Train-custom-Object-Detection-model/Fruits_for_detectron2/Validate")


# Extract the metadat and the dataset fictionaries :

train_metdata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

val_metdata = MetadataCatalog.get("my_dataset_val")
val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

# lets display the first train image with the annotations

from matplotlib import pyplot as plt 

first_dict = train_dataset_dicts[0]
print(first_dict)

filename = first_dict['file_name']
height = first_dict['height']
width = first_dict['width']
image_id = first_dict['image_id']
annotations = first_dict['annotations']

img = cv2.imread(filename)
visualizer = Visualizer(img[:, :, ::-1], metadata=train_metdata, scale=0.5)
vis = visualizer.draw_dataset_dict(first_dict)

img2 = vis.get_image()
img_rgb = cv2.cvtColor(img2 , cv2.COLOR_BGR2RGB)

cv2.imshow("img_rgb", img_rgb)
cv2.waitKey(0)




