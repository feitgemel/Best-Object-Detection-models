import os 
import cv2 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator , colors

model = YOLO("yolo11n.pt")
names = model.names 

# Motorcycle class id in Coco dataset is 3
motocross_id = 3

# Blur Ratio 
blue_ratio = 50 

cap = cv2.VideoCapture("Best-Object-Detection-models/Yolo-V11/Object Blurring using Ultralytics/motocross.mp4")
assert cap.isOpened(), "Error opening video file"
w , h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

crop_dir_name = "Best-Object-Detection-models/Yolo-V11/Object Blurring using Ultralytics/motocross_blur"
if not os.path.exists(crop_dir_name):
    os.makedirs(crop_dir_name)

# Video Writer 
video_writer = cv2.VideoWriter("Best-Object-Detection-models/Yolo-V11/Object Blurring using Ultralytics/motocross_blur.avi",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps,
                               (w, h))


idx = 0 
while cap.isOpened():
    success , im0 = cap.read()
    if not success:
        print("Error reading frame or Video frame is empty or Video proccesing has been completed.")
        break

    results = model.predict(im0, classes=[motocross_id], show=False)

    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    
    annotator = Annotator(im0, line_width=2, example=names)

    if boxes is not None:
        for box , cls in zip(boxes, clss):
            if cls == motocross_id:

                idx += 1
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)]) 

                obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                blur_obj = cv2.blur(obj, (blue_ratio, blue_ratio) )

                cv2.imwrite(os.path.join(crop_dir_name, str(idx) + ".png"), blur_obj) # Save the blur image

                im0[int(box[1]):int(box[3]), int(box[0]) : int(box[2] ) ] = blur_obj

    cv2.imshow("Original", im0)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
video_writer.release()
cv2.destroyAllWindows()
                

    


