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

# Define transformations
# This class will convert image to tensor before insert the data into the model :
class CocoTransform: 
    def __call__(self, image, target):
        image = F.to_tensor(image) # Convert PIL image to tensor 
        return image, target
    
# Dataset class -> for loading the data images 
def get_coco_dataset(img_dir , ann_file): # params : image folder and the annotation file 
    return CocoDetection(root=img_dir, annFile=ann_file, transforms=CocoTransform())


# Load datasets 
train_dataset = get_coco_dataset(
    img_dir = "D:/Data-Sets-Object-Detection/Tomato Leaf Disease.v63i.coco/train",
    ann_file = "D:/Data-Sets-Object-Detection/Tomato Leaf Disease.v63i.coco/train/_annotations.coco.json"
) 

val_dataset = get_coco_dataset(
    img_dir = "D:/Data-Sets-Object-Detection/Tomato Leaf Disease.v63i.coco/valid",
    ann_file = "D:/Data-Sets-Object-Detection/Tomato Leaf Disease.v63i.coco/valid/_annotations.coco.json"
) 


#Load the data:
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x))) 
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x))) 

print(f"Number of training images : {len(train_dataset)}")
print(f"Number of validation images : {len(val_dataset)}")

# ==================================================================

# Display the first image from the train dataset 

import cv2 
import numpy as np

# Get the first image
image , target = train_dataset[0] # First sample 

# Convert tensor to numpy array and reshape for OpenCV
image_np = image.permute(1,2,0).numpy()

# Normelaize 
image_np = (image_np * 255).astype(np.uint8)

# Convert to BGR format :
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Loop through all objects in the annotation 
for obj in target: # Traget is a list 
    bbox = obj["bbox"] # COCO format : [x , y, width, height]
    x, y, w, h = map(int , bbox) # Convert to integers 
    x_max , y_max = x+w , y+ h 

    # Draw bounding box
    cv2.rectangle(image_np, (x,y) , (x_max, y_max) , (0,255,0) ,2)

# Display the result image 
cv2.imshow("First train image with bounding boxes", image_np)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# Preprae the training ......
# Use the Faster R-CNN model based on Resnet-50 backbone 

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


# Init the model 
model = get_model(num_of_classes)

# Move the model to GPU if avaiable 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# Defince optimizer and learning rate scheduler 
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9 , weight_decay=0.0005)

# Help to schedule the learning rate to different epochs 
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3 , gamma=0.1)

# Train the model 
def train_one_epoch(model , optimizer , data_loader , device, epoch):

    # Set the model to train mode 
    model.train() 

    total_loss = 0 
    batch_count = 0
    skipped_batches = 0 
    total_boxes = 0 

    # Main progress bar for batch preocessing of images and its classes 
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}" , leave=True )

    # Iterate over the dataset 

    for images , targets in progress_bar:
        # Move images to the device (same as model)
        images = [img.to(device) for img in images]

        # Validate and process targets 
        processed_targets = []
        valid_images = [] 
        batch_boxes = 0 

        # target is bounding box and class labels
        # Prepare the data for the train -> The result is the list of "processed target" variable with boxes and labels for each image
        for i , target in enumerate(targets):
            boxes = [] 
            labels = []
            for obj in target :
                # Extract bbox 
                bbox = obj["bbox"] # Format : [x, y, width , height]
                x, y, w, h, = bbox 

                # Ensure the width and height are postive values 
                if w>0 and h>0 :
                    boxes.append([x, y, x+w , y+h ]) # Convert to [x_min, y_min , x_max, y_max]
                    labels.append(obj["category_id"])


            #Only process if there are valid boxes 
            if boxes:
                processed_target = {
                    "boxes":torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels":torch.tensor(labels, dtype=torch.int64).to(device),                    
                }
                processed_targets.append(processed_target)
                valid_images.append(images[i]) # Add only valid images 
                batch_boxes += len(boxes)

            else :
                # Mention skipped images in progress bar description 
                progress_bar.set_postfix_str(f"Skipping image {i} (no valid boxes)") 

        # Skip iteration if no valid targets 
        if not processed_targets:
            skipped_batches +=1 
            progress_bar.set_postfix_str("Skipping batch - No valid targets")

        # Ensure images and targets are aligned 
        images = valid_images
        total_boxes += batch_boxes 

        # This is the train process ( images + bounding boxes + labels)
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        # Back propagation 
        optimizer.zero_grad() 
        losses.backward() 
        optimizer.step() 

        total_loss += losses.item() 
        batch_count += 1

        # Update progress bar with current loss and other metrics 
        progress_bar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'avg_loss': f'{total_loss/batch_count:.4f}',
            'boxes': total_boxes,
            'skipped': skipped_batches
        })


    # Calculate average loss for epoch 
    avg_loss = total_loss / max(batch_count , 1)

    # Print Epoch summary 
    tqdm.write(f"\n[Epoch {epoch+1}] "
               f"Processed: {batch_count}/{len(data_loader)} batches "
               f"({skipped_batches} skipped) | "
               f"Total boxes: {total_boxes} | "
               f"Average loss: {avg_loss:.4f}")
    

    return avg_loss



# Run the process for 5 epochs 
losses = [] 
num_epochs = 5 
epoch_bar = tqdm(range(num_epochs), desc="Training Progress", position=0)

# Modified training loop with Tqdm 
print("======== Start Training ========")
print(f"Model: Faster R-CNN with Resnet-50 backbone ")
print(f"Number of classes : {num_of_classes}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Number of epochs : {num_epochs}")
print("==================================")



for epoch in epoch_bar:
    avg_loss = train_one_epoch(model , optimizer , train_loader, device, epoch)
    losses.append(avg_loss)
    lr_scheduler.step() 

    # Save the model state dictionary after every epoch 
    model_save_dir = "d:/temp/models/Tomato-Leaf-Disease"
    os.makedirs(model_save_dir, exist_ok=True)

    model_path = f"d:/temp/models/Tomato-Leaf-Disease/Faster_rcnn_resnet50_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict() , model_path)


    # Update epoch progress bar 
    epoch_bar.set_postfix({
        'current_loss': f'{avg_loss:.4f}',
        'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
    })
    tqdm.write(f"Model saved: {model_path}")


# Print final training summary 
print("\n ================== Training Complete ===================")
print(f"Loss progression across epochs: {[f'{loss:.4f}' for loss in losses]}")





