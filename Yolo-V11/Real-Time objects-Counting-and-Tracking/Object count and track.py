import cv2 
from ultralytics import YOLO 
from collections import defaultdict 

# Load the Yolo model 
model = YOLO("yolo11l.pt")

class_list = model.names
print(class_list)

# Open the video file :
cap = cv2.VideoCapture("Best-Object-Detection-models/Yolo-V11/Real-Time objects-Counting-and-Tracking/Horses.mp4")

# Red line position
line_y_red = 600 

# Dictionary to store object counts by class
class_counts = defaultdict(int)

# Dictionary to keep track of ibject IDs that have crossed the line 
crossed_ids = set() 

# Look through the video frames 
while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break

    # Run Yolo tracking on the frame 
    results = model.track(frame, persist=True, classes=[17], conf=0.6) # Only horeses with confidence above 60%

    # Ensure results are not empty 
    if results[0].boxes.data is not None:
        # get the detected boxes , and their classes indices and track id 
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        # Put a red line for count crossing 
        cv2.line(frame, (200, line_y_red),(1500, line_y_red), (0,0,255) , 3)

        # Loop through each detected object and extract the bounding boxes 
        for box, track_id , class_idx , conf in zip(boxes, track_ids, class_indices, confidences):

            x1, y1 , x2, y2 = map(int, box)

            # Claculate the center point 
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            class_name = class_list[class_idx]

            # Put a red dot in the center of the object 
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1 ) 

            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) 
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Check if the object has crossed the red line 
            if cy > line_y_red and track_id not in crossed_ids:
                # Mark the object as crossed 
                crossed_ids.add(track_id)
                class_counts[class_name] += 1

        # Display the counts on the frame 
        y_offset = 150 
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
            y_offset += 30 
    

    # Display the frame 
    cv2.imshow("YoloV11 object tracking & counting", frame)

    # Exit loop if 'q' key is pressed 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

