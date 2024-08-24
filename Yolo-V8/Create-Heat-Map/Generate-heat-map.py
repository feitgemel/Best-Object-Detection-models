from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2 

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("Best-Object-Detection-models/Yolo-V8/Create-Heat-Map/dogs.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("Best-Object-Detection-models/Yolo-V8/Create-Heat-Map/heatmap_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w,h))

# Init heatmap 
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                     imw=w,
                     imh=h,
                     view_img=True,
                     shape="circle")

while cap.isOpened():
    success , frame = cap.read()
    if not success:
        print("Video frame is empty or completed.")

    tracks = model.track(frame, persist=True, show=False)

    im0 = heatmap_obj.generate_heatmap(frame, tracks)

    cv2.imshow("im0", im0)
    if cv2.waitKey(1) == ord('q') :
        break

    video_writer.write(im0)


cap.release()
video_writer.release()
cv2.destroyAllWindows()