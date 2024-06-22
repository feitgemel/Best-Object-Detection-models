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


# Train the model with Pytorch Lightinig
# The DETR model is loading using the Hugging Face Transformer library

import pytorch_lightning as pl 
from transformers import DetrForObjectDetection 
import torch 

CHECKPOINT = "facebook/detr-resnet-50"

# This is the main class for our model :

class Detr(pl.LightningModule):

    def __init__(self, lr , lr_backbone, weight_decay):
        super().__init__()
        # loading the pre trained model 
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )

        self.lr = lr 
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    # Forward pass
    def forward(self , pixel_values, pixel_mask ):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    # Provide the loss and loss dictionary
    # We use this function for  the train and the validation as well

    def common_step(self , batch, batch_idx):
        pixel_values = batch["pixel_values"] 
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k,v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values , pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss 
        loss_dict = outputs.loss_dict 

        return loss , loss_dict
    
    def training_step(self , batch , batch_idx):
        loss , loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training step , and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("Train_" + k, v.item())
        
        return loss

    def validation_step(self , batch , batch_idx):
        loss , loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training step , and the average across the epoch
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("Valiation_" + k, v.item())
        
        return loss
    
    # Define the optimizer and the learning rate
    def configure_optimizers(self):

        param_dicts = [
            {
                "params": [p for n,p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr= self.lr , weight_decay=self.weight_decay)
    

    # load the train data
    def train_dataloader(self):
        return TRAIN_DATALOADER
    
    #load the validation data 
    def val_dataloader(self):
        return VAL_DATALOADER
    

# train the model 

# lr -> Learning rate
# lr_backbone -> for extracing the CNN features
# weaight_decay -> relevant for the optimizer

# Create the model object 
model = Detr(lr = 1e-4 , lr_backbone=1e-5 , weight_decay=1e-4)

# Train the model
from pytorch_lightning import Trainer 
log_dir = 'C:/temp/my_DETR_log'

MAX_EPOCHS = 200

trainer = Trainer(devices=1,
                  accelerator="gpu",
                  max_epochs=MAX_EPOCHS,
                  gradient_clip_val=0.1 , 
                  accumulate_grad_batches=8 ,
                  log_every_n_steps=1,
                  default_root_dir=log_dir)

trainer.fit(model)

# create a saved folder :
MODEL_PATH = "C:/temp/DETR-My-Model-1"
model.model.save_pretrained(MODEL_PATH)






























































































