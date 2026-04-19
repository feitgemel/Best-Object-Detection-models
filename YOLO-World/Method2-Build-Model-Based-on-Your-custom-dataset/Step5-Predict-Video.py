from ultralytics import YOLOWorld
import cv2 
import os

video_path = "D:/Data-Sets-Object-Detection/trash_ICRA19/dataset/videos_for_testing/manythings.mp4"
output_path = "D:/Data-Sets-Object-Detection/trash_ICRA19/dataset/videos_for_testing/output_video.mp4"

# Load the model 
model_path = os.path.join("D:/Temp/Models/Underwater-Trash", "My-YoloWorld-Model-S","weights", 'best.pt')

# Load the model 
model = YOLOWorld(model_path)
threshold = 0.25 

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define Video Writer to the save output 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict
    imgPredict = frame.copy()
    results = model(imgPredict)[0]

    # Draw the bounding boxes 
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(imgPredict, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(imgPredict, results.names[int(class_id)].upper(), (int(x1), int(y1 + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0), 1) 
            
    # Show frane 
    cv2.imshow("Prediction", imgPredict)
    out.write(imgPredict)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
