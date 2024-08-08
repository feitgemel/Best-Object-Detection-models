import cv2 
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")

# Run inference
img = cv2.imread("Best-Object-Detection-models/Yolo-Nas/Simple-Object-Detection-using-Yolo-Nas/haverim.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

results = model.predict(img)

print(results)

results.show()

# Another way to display the detected objects :

bboxes = results.prediction.bboxes_xyxy # Bounding boxes in (x1, y1, x2, y2) format
labels = results.prediction.labels # Class IDs of the detected objects 
confidences = results.prediction.confidence # Confidence scores of the detections
class_names = results.class_names # List of all class names

# Draw detections on the image 

for bbox , label , confidence in zip(bboxes, labels, confidences):
    x1, y1, x2, y2 = map(int , bbox)
    label_text = f"{class_names[label]} {confidence:.2f}"

    # Draw a bounding box 
    cv2.rectangle(img , (x1,y1), (x2,y2), (0,0,255), 2)

    #put the label above the boudnding box
    cv2.putText(img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

# convert the img to BGR
output_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("c:/temp/detected.png", output_img)

# Display 
cv2.imshow("Detected image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


