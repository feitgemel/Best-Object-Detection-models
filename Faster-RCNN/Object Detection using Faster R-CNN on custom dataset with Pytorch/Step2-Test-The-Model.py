import torch 
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN 
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F 
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import cv2 
import os 

def get_model(num_classes):
    # load pre-trained Faster R-CNN 
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Get the number of input features for the classifer 
    # We would like to use the pre-trained model weights 
    in_featurs = model.roi_heads.box_predictor.cls_score.in_features 

    # Replace tje pre-trained head with a new one 
    # Change the final data layer with out layer ! 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_featurs,num_classes)
    return model 


classes = [
    "Tomato-Leaf",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Miner",
    "Leaf Mold",
    "Mosaic Virus",
    "Septoria",
    "Spider Mites",
    "Yellow Leaf Curl Virus"]

num_of_classes = len(classes) + 1 # Add backgournd class

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the saved model 
model_path = f"d:/temp/models/Tomato-Leaf-Disease/Faster_rcnn_resnet50_epoch_5.pth"
model = get_model(num_of_classes)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval() 
 
# === Load image
def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    return image_tensor.to(device)


test_path = "D:/Data-Sets-Object-Detection/Tomato Leaf Disease.v63i.coco/test"
annotation_path = "D:/Data-Sets-Object-Detection/Tomato Leaf Disease.v63i.coco/test/_annotations.coco.json"

import json
import random


# Load COCO annotations 
with open(annotation_path, "r") as f:
    coco_annotations = json.load(f)

# Create image Id to filname mapping 
image_id_to_filename = {img['id']: img['file_name'] for img in coco_annotations['images']}
image_filename_to_id = {filename: img_id for img_id, filename in image_id_to_filename.items()}

# Get a list of all images in the test directory 
all_images = [img for img in os.listdir(test_path) if img.endswith((".jpg", ".png"))] 

# Select 6 ranfom images 
random_images = random.sample(all_images, 6)

# Convert the list to dictionay 
COCO_CLASSES = {
    i + 1: cls
    for i , cls in enumerate(classes)
}    

COCO_CLASSES[0]= "Background"

# Draw bounding annotations 


def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")

def get_ground_truth_annotations(filename):
    # Find the image id for the given filename
    image_id = image_filename_to_id.get(filename)
    
    # Find all annotations for this image
    ground_truth_annotations = []
    for annotation in coco_annotations['annotations']:
        if annotation['image_id'] == image_id:
            # COCO format: [x, y, width, height]
            # Convert to [x_min, y_min, x_max, y_max]
            x, y, width, height = annotation['bbox']
            ground_truth_annotations.append({
                'bbox': [x, y, x + width, y + height],
                'category_id': annotation['category_id']
            })
    
    return ground_truth_annotations

# Function to draw bounding boxes
def draw_boxes(ax, image, prediction, ground_truth):
    ax.imshow(image)
    
    # Draw ground truth boxes in blue
    for gt_box in ground_truth:
        x_min, y_min, x_max, y_max = gt_box['bbox']
        class_name = get_class_name(gt_box['category_id'])
        text = f"True - {class_name}"
        
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                   linewidth=2, edgecolor='b', facecolor='none'))
        ax.text(x_min, y_min-20, text, color='blue', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Draw predicted boxes in red
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    threshold = 0.5

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)
            text = f"Predicted - {class_name} ({score:.2f})"
            
            # Draw bounding box
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                       linewidth=2, edgecolor='r', facecolor='none'))
            # Display text with white background
            ax.text(x_min, y_min, text, color='red', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    ax.axis('off')

# Run the inference for 6 random images :
fig , axes = plt.subplots(2, 3, figsize=(20,14) ) 

for ax , img_name in zip(axes.flatten() , random_images):
    img_path = os.path.join(test_path , img_name)
    image_tensor = prepare_image(img_path)
    
    # Get the ground truth annotations 
    ground_truth = get_ground_truth_annotations(img_name) 

    with torch.no_grad():
        prediction = model(image_tensor)

    image = Image.open(img_path)
    draw_boxes(ax, image, prediction, ground_truth)

plt.tight_layout()
plt.show()

