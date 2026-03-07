import cv2 
import numpy as np

# Load image :

imagePath = "Best-Object-Detection-models/Media-Pipe/the-last-of-us.jpg"
img = cv2.imread(imagePath)
cv2.imshow("Original Image", img)
#cv2.waitKey(0)

# Step 1 import the neccesary mediapipe modules 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Step 2 create an object detector and set the options
base_options = python.BaseOptions(model_asset_path='D:/Temp/Models/MediaPipe/efficientdet_lite2.tflite') 
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.4) 
detector = vision.ObjectDetector.create_from_options(options)

# Step 3 - Load the imput image 
image = mp.Image.create_from_file(imagePath)

# Step 4 - Run object detection on the input image
detection_result = detector.detect(image)

print('Detection result: {}'.format(detection_result))

# Step 5 : Process the detection result and visualize it on the input image
image_with_detected_objects = np.copy(image.numpy_view())
                                      
TEXT_COLOR = (255,0,0) # Red
MARGIN = 10 # piexels
ROW_SIZE = 10 # pixels
FONT_SIZE = 1 # font size
FONT_THICKNESS = 1 # font thickness

for detection in detection_result.detections:
    # Draw bounding box
    bbox = detection.bounding_box
    start_point = (bbox.origin_x, bbox.origin_y)
    end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
    cv2.rectangle(image_with_detected_objects, start_point, end_point,(TEXT_COLOR), 3) 

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)

    cv2.putText(image_with_detected_objects, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

image_with_detected_objects = cv2.cvtColor(image_with_detected_objects, cv2.COLOR_RGB2BGR)
cv2.imshow('Object Detection Result', image_with_detected_objects)
cv2.imwrite('Best-Object-Detection-models/Media-Pipe/the-last-of-us-detected.jpg', image_with_detected_objects) 
cv2.waitKey(0)

