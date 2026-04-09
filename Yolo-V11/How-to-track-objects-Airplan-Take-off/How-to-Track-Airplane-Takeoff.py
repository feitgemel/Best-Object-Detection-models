import cv2 
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11l.pt")
video_path = "Best-Object-Detection-models/Yolo-V11/How-to-track-objects-Airplan-Take-off/Airplane.mp4"

cap = cv2.VideoCapture(video_path)

box_annotator = sv.BoxAnnotator(thickness=2) 
label_annotator = sv.LabelAnnotator()

tracker = sv.ByteTrack()

while cap.isOpened():
    
    success , frame = cap.read()
    if not success:
        break

    # Run tracking
    results = model.track(frame, persist=True, show=False)[0]

    # Convert to Supervision detections

    detections = sv.Detections.from_ultralytics(results)  
    detections = tracker.update_with_detections(detections)

    # Build the labels 
    labels = [f"{tracker_id}" for tracker_id in detections.tracker_id] if detections.tracker_id is not None else [""]

    # Annotate 
    annotate_frame = box_annotator.annotate(scene=frame.copy(), detections=detections) 
    annotate_frame = label_annotator.annotate(scene=annotate_frame, detections=detections, labels=labels)

    cv2.imshow("Yolo tracking ", annotate_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

