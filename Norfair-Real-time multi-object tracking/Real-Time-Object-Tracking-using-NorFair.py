import cv2 
from ultralytics import YOLO
from vidgear.gears import CamGear
from norfair import Detection, Tracker, draw_tracked_objects
import numpy as np

# Path to the Youtube video
stream = CamGear(source="https://www.youtube.com/watch?v=msn0zfdEk58", stream_mode=True, logging=True).start()


# Load the Yolo model 
model = YOLO('yolov8n.pt') # Load the Yolo V8 nano model
threshold = 0.25 
person_class_id = 0 # YoloV8 class Id for "person"


# Init Norfair tracker 
tracker = Tracker(distance_function="euclidean", distance_threshold=100)


# Main loop 

while True:
    # Read frames from the video stream
    frame = stream.read()

    if frame is None:
        break

    #Run YoloV8 detection for each frame 
    detections = model(frame)
    results = detections[0]

    # Convert detections to Norfair format 
    norfair_detections = [Detection(points=np.array([(box[0] + box[2]) /2 , (box[1] + box[3]) /2 ])) 
                          for box in results.boxes.data.tolist() if box[4] > threshold and int(box[5]) == person_class_id]
    
    # Update tracker with new detections
    tracked_objects = tracker.update(detections=norfair_detections)

    # Draw tracked objects on the frame
    draw_tracked_objects(frame , tracked_objects)

    # Display bounding boxes for people with confidnce scored above the thereshold
    for result in results.boxes.data.tolist():
        x1, y1 , x2, y2, score, class_id = result 

        if score > threshold and int(class_id) == person_class_id:
            cv2.rectangle(frame , (int(x1), int(y1)) , (int(x2), int(y2)) , (0,255,0) ,2)
            cv2.putText(frame , results.names[int(class_id)].upper(), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0) ,1 , cv2.LINE_AA)


    # Display the processed frame
    cv2.imshow("Video Stream", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
stream.stop()
cv2.destroyAllWindows()

