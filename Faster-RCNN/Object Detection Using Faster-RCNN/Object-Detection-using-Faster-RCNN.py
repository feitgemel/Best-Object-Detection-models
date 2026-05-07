from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
object_detection_model = fasterrcnn_resnet50_fpn(weights=weights, progress=True) 

# Setting the model for run evaluation 
object_detection_model.eval()


# Display the image : 
from PIL import Image
test_img = Image.open("Best-Object-Detection-models/Faster-RCNN/Object Detection Using Faster-RCNN/train.jpg")

test_img.show() 



# Predict an image :

# Object Detection - Step 1
# Faster R-CNN (Resnet 50 Backbon) | High Precision


import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
]) 

# Apply the transformation to the image :
test_img_tensor = transform(test_img)

print(test_img_tensor) 
print(test_img_tensor.shape )

# Add batch dimataion -> Create a batch of images that contain one image (for the model)

test_img_tensor = test_img_tensor.unsqueeze(dim=0)

print("Add batch dimension : ")
print(test_img_tensor.shape)

# Run the predicition :
preds = object_detection_model(test_img_tensor)

print("Predictions Results : ")
print(preds)


# LOAD THE COCO.NAMES
with open("Best-Object-Detection-models/Faster-RCNN/Object Detection Using Faster-RCNN/coco.names","r") as f:
    COCO_INSTANCE_CATEGORY_NAMES = f.read().splitlines()

print(COCO_INSTANCE_CATEGORY_NAMES)

# Visualization with the class names :
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Convert tensor image back to numpy for visualization 
img = test_img_tensor.squeeze().detach().numpy() # Adjust dimentions as necessary 
img = img.transpose((1, 2, 0)) # Rearrage the dimentions from CxHxW to HxWxC for image display 

# Create figure and axes 
fig, ax = plt.subplots(1, figsize=(12, 9 )) 

# Display the image 
ax.imshow(img)

CONFIDENCE_THRESHOLD = 0.5

# Add bounding boxes and labels 
for box , label , score in zip(preds[0]['boxes'], preds[0]['labels'], preds[0]['scores']):
    if score > CONFIDENCE_THRESHOLD :
        x1 , y1 , x2 , y2 = box.detach().cpu().numpy() 
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f'{label_name} {score.item():.2f}' , color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5)) 

plt.axis('off')
plt.show()

# 2. Using SSD with Mobilenet backbone | Optimized for speed 

from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT 
object_detection_model = ssdlite320_mobilenet_v3_large(weights=weights, progress=True) 

object_detection_model.eval() # Setting model for prediction
preds = object_detection_model(test_img_tensor)

print(preds) 

# Convert tensor image back to numpy for visualization 
img = test_img_tensor.squeeze().detach().numpy() # Adjust dimentions as necessary 
img = img.transpose((1, 2, 0)) # Rearrage the dimentions from CxHxW to HxWxC for image display 

# Create figure and axes 
fig, ax = plt.subplots(1, figsize=(12, 9 )) 

# Display the image 
ax.imshow(img)

CONFIDENCE_THRESHOLD = 0.5

# Add bounding boxes and labels 
for box , label , score in zip(preds[0]['boxes'], preds[0]['labels'], preds[0]['scores']):
    if score > CONFIDENCE_THRESHOLD :
        x1 , y1 , x2 , y2 = box.detach().cpu().numpy() 
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f'{label_name} {score.item():.2f}' , color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5)) 

plt.axis('off')
plt.show()






