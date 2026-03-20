import cv2 
import supervision as sv 
from ultralytics import YOLO
import numpy as np 

model = YOLO("yolov8m.pt") 

box_annotator = sv.BoxAnnotator() 
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture("Best-Object-Detection-models/SuperVision-Object-Detection/test.mp4")
w,h,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS )   )

out = cv2.VideoWriter("Best-Object-Detection-models/SuperVision-Object-Detection/output2.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w,h) ) 

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


    # Round box annotator :
    round_box_annotator = sv.RoundBoxAnnotator() 
    finalImage2 = round_box_annotator.annotate(scene=img.copy(), detections=detections) 
    

    # Box Corner Annotator
    corner_annotator = sv.BoxCornerAnnotator() 
    finalImage3 = corner_annotator.annotate(scene=img.copy(), detections=detections) 


    # Color Annotator 
    color_annotator = sv.ColorAnnotator() 
    finalImage4 = color_annotator.annotate(scene=img.copy(), detections=detections) 


    # Circle Annotator 
    circle_annotator = sv.CircleAnnotator() 
    finalImage5 = circle_annotator.annotate(scene=img.copy(), detections=detections)

    #Traingle Annotator 
    triangle_annotator = sv.TriangleAnnotator() 
    finalImage6 = triangle_annotator.annotate(scene=img.copy(), detections=detections)

    # Blur annotator 
    blur_annotator = sv.BlurAnnotator() 
    finalImage7 = blur_annotator.annotate(scene=img.copy(), detections=detections)
                                          
 
    out.write(finalImage1) 

    cv2.imshow("Original", img )
    cv2.imshow("finalImage1", finalImage1 )
    cv2.imshow("finalImage2", finalImage2 )
    cv2.imshow("finalImage3", finalImage3 )
    cv2.imshow("finalImage4", finalImage4 )
    cv2.imshow("finalImage5", finalImage5 )
    cv2.imshow("finalImage6", finalImage6 )
    cv2.imshow("finalImage7", finalImage7 )


    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break 


cv2.destroyAllWindows()
cap.release()
out.release()


