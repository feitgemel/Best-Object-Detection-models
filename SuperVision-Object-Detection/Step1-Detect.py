import cv2 
import supervision as sv 
from ultralytics import YOLO
import numpy as np 

model = YOLO("yolov8m.pt") 

box_annotator = sv.BoxAnnotator() 
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture("Best-Object-Detection-models/SuperVision-Object-Detection/test.mp4")
w,h,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS )   )

out = cv2.VideoWriter("Best-Object-Detection-models/SuperVision-Object-Detection/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w,h) ) 

# Optional - Display only selectec classes -> dogs
selected_classes = [16]

while cap.isOpened(): 
    ret, img = cap.read() 

    if not ret: 
        break

    
    # Predicitions :
    result = model(img)

    detections = sv.Detections.from_ultralytics(result[0])

    # Optional step - filter classes for only dogs 
    #detections = detections[np.isin(detections.class_id, selected_classes)]

    # Filter only confidence above 0.5
    detections = detections[detections.confidence > 0.5]

    labels = [model.model.names[class_id] for class_id in detections.class_id]
    
    finalImage1 = box_annotator.annotate(scene=img.copy(), detections=detections) 
    finalImage1 = label_annotator.annotate(scene=finalImage1, detections=detections, labels=labels) 

    out.write(finalImage1) 

    cv2.imshow("Original", img )
    cv2.imshow("Output", finalImage1 )


    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break 


cv2.destroyAllWindows()
cap.release()
out.release()


