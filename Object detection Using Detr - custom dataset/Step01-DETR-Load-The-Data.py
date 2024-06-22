import torch 
import supervision as sv 
import transformers 
import pytorch_lightning
import os 
import torchvision

# Link for the dataset : https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg
# Download the dataset with COCO format

dataset = "E:/Data-sets/bone fracture.v2-release.coco" # dataset folder

ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset , "train")
VAL_DIRECTORY = os.path.join(dataset , "valid")
TEST_DIRECTORY = os.path.join(dataset , "test")

# Class for load the dataset 

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__( self, image_directory_path: str, image_processor, train: bool = True):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self , idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images = images , annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["lables"][0]

        return pixel_values, target


# lets create 3 data objects

from transformers import DetrImageProcessor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor,
                              train=True)

VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor,
                              train=False)

TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor,
                              train=False)

print("Number of train images :", len(TRAIN_DATASET))
print("Number of validation  images :", len(VAL_DATASET))
print("Number of test images :", len(TEST_DATASET))

# visual the images 
import random 
import cv2 
import numpy as np 

# Select a random image

image_ids = TRAIN_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
print('Image #{}'.format(image_id))

# load the image with annotation
image = TRAIN_DATASET.coco.loadImgs(image_id)[0]
annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])
image = cv2.imread(image_path)

# Annotate 
detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

categories = TRAIN_DATASET.coco.cats 
print("categories:" , categories)

id2label = {}
for k , v in categories.items():
    id2label[k] = v['name']

labels = [] 

for _,_, class_id , _ in detections:
    labels.append(f"{id2label[class_id]}")

print("==================================================================")
print("id2label", id2label)
print("labels", labels)

box_annotator = sv.BoxAnnotator()
frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
cv2.imshow("Image", image)
cv2.waitKey(0)












 
