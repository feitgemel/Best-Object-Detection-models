import cv2 
import time 
import os 
import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import get_file

# Step1 - Load classes and attach colors to each class

np.random.seed(100) 
classFilePath = "Best-Object-Detection-models/Object Detection with 40 Models/coco.names"

with open(classFilePath, 'r') as f:
    classesList = f.read().splitlines()

colorList = np.random.uniform(low=0, high=255, size=(len(classesList), 3))

print("Total Number of Classes Detected: ", str(len(classesList))) 
print(classesList)

print("Color List: ", colorList)

# Step2 - Load the model

# Here is the list of the 40 supported models.
# trained for the COCO dataset
#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

modelUrl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz"
#modelUrl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

fileName = os.path.basename(modelUrl)
print("File Name: ", fileName)

onlyFileName = fileName[:fileName.index('.')]
print("Only File Name: ", onlyFileName)

# Download the model and extract it
saveFolder = "/mnt/d/Temp/Models/40Models"
os.makedirs(saveFolder, exist_ok=True)

# Download the model
get_file(fname = onlyFileName,
         origin = modelUrl,
         cache_dir = saveFolder,
         cache_subdir = "checkpoints",
         extract = True)

print("Model Downloaded and Extracted Successfully")



# Step 3 - Load the model into memory
print("Loading the model into memory...")

tf.keras.backend.clear_session()
fullPath = os.path.join(saveFolder, "checkpoints", onlyFileName, onlyFileName, "saved_model")
print("Full Path: ", fullPath)

model = tf.saved_model.load(fullPath)

print("Model "+ onlyFileName + " loaded successfully")

# Step 4 = Predict Image 

threshold = 0.5
imagePath = "Best-Object-Detection-models/Object Detection with 40 Models/Inbal-Midbar 768.jpg" 
#imagePath = "Best-Object-Detection-models/Object Detection with 40 Models/Dori.jpg"

original_image = cv2.imread(imagePath)
image = original_image.copy()

inputTensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to tensor
inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)

# Create batch of images (in our case only one image in the batch)
inputTensor = inputTensor[tf.newaxis, ...]

# Get detections 
detections = model(inputTensor)

print("Detections: ", detections)

bboxes = detections['detection_boxes'][0].numpy()
classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
ClassScores = detections['detection_scores'][0].numpy()

H , W, C = image.shape

# Reduce the overlap of the bounding boxes using Non-Maximum Suppression
bboxIdx = tf.image.non_max_suppression(bboxes, ClassScores, max_output_size=50, # maximum number of boxes to be selected
                                       iou_threshold= threshold, # 50% overlap
                                       score_threshold= threshold) # minimum score

print("bboxIdx: ", bboxIdx)


# Display the results based on the reduced bounding boxes

if len(bboxIdx) != 0:
    for i in bboxIdx:
        bbox = tuple(bboxes[i].tolist())
        classConfidence = round(100 * ClassScores[i] )
        classIndex = classIndexes[i]

        classLabelText = classesList[classIndex] 
        classColor = colorList[classIndex]

        displayText = "{}: {}%".format(classLabelText, classConfidence)

        ymin, xmin, ymax, xmax = bbox
        print("ymin: ", ymin, "xmin: ", xmin, "ymax: ", ymax, "xmax: ", xmax)

        xmin, xmax, ymin, ymax = (xmin * W, xmax * W, ymin * H, ymax * H)

        xmin , xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1) 
        cv2.putText(image, displayText, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, thickness=2)

cv2.imshow("Original", original_image)
cv2.imwrite("Best-Object-Detection-models/Object Detection with 40 Models/Output.jpg", image)

cv2.imshow("Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()


















































