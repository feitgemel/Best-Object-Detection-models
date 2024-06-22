import torch 
import supervision as sv 
import transformers 
import pytorch_lightning
import os 
import torchvision

# Link for the dataset : https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg
# Download the dataset with COCO format

dataset = "c:/Data-sets/bone fracture.v2-release.coco" # dataset folder

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
        target = encoding["labels"][0]

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


# Prepare the data before training
# ================================

# collate_fn is a function that defines how the samples (in a batch) should be processed before being passed to the model 

def collate_fn(batch):
    pixel_vales = [] 

    for item in batch:
        pixel_vales.append(item[0])

    # The images should be in the same size - Using the pad function

    encoding = image_processor.pad(pixel_vales, return_tensors="pt")

    labels = []
    for item in batch :
        labels.append(item[1])

    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

from torch.utils.data import DataLoader 
torch.set_float32_matmul_precision('medium')

categories = TRAIN_DATASET.coco.cats 
print("Categories:")
print(categories)

id2label = {}
for k, v in categories.items():
    id2label[k] = v['name']

print("id2label :")
print(id2label)
print(len(id2label))

print("=====================================================")
TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET,
                              collate_fn=collate_fn,
                              batch_size=4,
                              shuffle=True)

VAL_DATALOADER = DataLoader(dataset=VAL_DATASET,
                              collate_fn=collate_fn,
                              batch_size=4)

import pytorch_lightning as pl 
from transformers import DetrForObjectDetection 
import torch 


MODEL_PATH = "C:/temp/DETR-My-Model-1"
device = "cuda"
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(device)

print(model)

# ================================================================================

box_annotator = sv.BoxAnnotator()

import cv2 

# load image fro the Test Folder
#image_path = "C:/Data-sets/bone fracture.v2-release.coco/test/16_jpg.rf.0ef960d157f3f332421d9d5a8248a00f.jpg"
image_path = "C:/Data-sets/bone fracture.v2-release.coco/test/117_jpg.rf.119dccd2483b04d8d3a8c33a1393d362.jpg"



image = cv2.imread(image_path)

CONFIDENCE_THRESHOLD = 0.35 
# Predict the bounding box annotations using our model

with torch.no_grad():

    # load the image and predict 
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs) # run the prediction

    # post process to show the image with the predicitions
    target_sizes = torch.tensor([image.shape[:2]]).to(device)
    results = image_processor.post_process_object_detection(
        outputs = outputs,
        threshold=CONFIDENCE_THRESHOLD,
        target_sizes=target_sizes)[0]

# create the output image 
detections = sv.Detections.from_transformers(transformers_results=results)
labels = [f"{id2label[class_id]} {confidence:.2f}" for _,confidence , class_id, _ in detections]
image_with_detection = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

cv2.imwrite("predict.png", image_with_detection )

cv2.imshow("image with detections", image_with_detection)
cv2.imshow("img", image)
cv2.waitKey(0)
