import cv2 
from ultralytics import YOLO
import os 
from vidgear.gears import CamGear 

# path to the video file
test_url = 'https://youtu.be/7qKU1b2Shr8?si=XWTU1Fbc0XtIs-yv'
model_path = os.path.join("C:/Data-sets/Mac-Real/dataset/checkpoints","small-Model","weights","best.pt")

# load the model
model = YOLO(model_path)
threshold = 0.25 
n= 0 

# path to our Youtube video file
stream = CamGear(source=test_url, stream_mode= True, logging=True).start() 

while True :

    frame = stream.read()
    if frame is None:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2 , score, class_id = result 

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        if score > threshold :
            cv2.rectangle(frame , (x1,y1), (x2,y2), (3,240,252), 1)
            cv2.putText(frame , results.names[int(class_id)].upper(), (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3,240,252), 1)
            

    cv2.imshow('img', frame)
    if cv2.waitKey(25) & 0XFF == ord('q'):
        break

cv2.destroyAllWindows()
stream.stop()


