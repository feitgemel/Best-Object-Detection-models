import cv2 
from ultralytics import YOLO 
import os 
from vidgear.gears import CamGear

# path to our video file :
stream = CamGear(source="https://www.youtube.com/watch?v=3sgewysRGZY", stream_mode=True , logging=True ).start()

# load the model 
model = YOLO('yolov8n.pt') # load the YoloV8 nano model 
threshold = 0.25

# Read the video and grab the frames :

while True:
    frame = stream.read()

    # check the frame :
    if frame is None:
        break # break the loop if the frame is not OK or finish the video 

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2 , score, class_id = result 

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        if score > threshold:
            cv2.rectangle(frame , (x1, y1), (x2,y2), (0,255,0), 1)
            cv2.putText(frame , results.names[int(class_id)].upper() , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    
    # Displat the frame 
    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
stream.stop()

