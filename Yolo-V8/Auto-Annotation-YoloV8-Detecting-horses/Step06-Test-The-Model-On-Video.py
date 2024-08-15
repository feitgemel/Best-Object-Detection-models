import cv2 
from ultralytics import YOLO 
import os 

# setup paths
video_path = "C:/Data-sets/Horse-race/Source-Data/Test-videos/video12.mp4"
model_path = os.path.join("C:/Data-sets/Horse-race/dataset/checkpoints","My-Large-Model","weights","best.pt")

# load the model
model = YOLO(model_path)
threshold = 0.25 

# Create a video capture obejct to read the video
cap = cv2.VideoCapture(video_path)

# Read and disply the video frame bu frame 
while True:
    ret , frame = cap.read() 

    # Check if the frame is successfully read
    if not ret:
        break 

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1 , x2, y2, score, class_id = result 

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        if score > threshold:
            cv2.rectangle(frame , (x1, y1), (x2, y2), (0,255,0), 1)
            cv2.putText(frame , results.names[int(class_id)].upper() , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0), 1)

    # Displat the frmae
    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF ==ord('q'):
        break 

# Release the Video capture and closee all the cv2 opened winodws
cap.release()
cv2.destroyAllWindows()