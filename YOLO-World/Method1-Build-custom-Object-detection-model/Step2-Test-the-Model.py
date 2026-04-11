from ultralytics import YOLO # Yolo and not YoloWorld
import cv2

pathForSavedModel = "d:/temp/models/my-yolov8s-world.pt"

# load your custom model 

model = YOLO(pathForSavedModel)

# Run inference to detect your cusom classes
testImage = "Best-Object-Detection-models/YOLO-World/Method1-Build-custom-Object-detection-model/test_image.jpg"


results = model.predict(testImage, device='cuda', conf=0.6) 

# Option 1 - Display the results 
results[0].show()


# Option 2 - Extract and display the results :

import supervision as sv
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=8, 
                                    text_thickness=10,
                                    text_color=sv.Color.BLUE,
                                    color=sv.Color.YELLOW) 

xyxy = results[0].boxes.xyxy.cpu().numpy() # Get the bounding boxes
confidence = results[0].boxes.conf.cpu().numpy() # Get the confidence score
class_id = results[0].boxes.cls.cpu().numpy().astype(int) # Get the class id

# Create a detection object 
detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

image = cv2.imread(testImage)

labels = [model.model.names[class_id] for class_id in detections.class_id]

print(labels)

# Draw bounding box
finalImage1 = box_annotator.annotate(scene=image.copy(), detections=detections)
 
# Draw the labels
finalImage1 = label_annotator.annotate(scene=finalImage1, detections=detections, labels=labels)

import cv2

# Save using OpenCV
cv2.imwrite("d:/temp/test_image.png", finalImage1)

sv.plot_image(finalImage1, (10,10))






