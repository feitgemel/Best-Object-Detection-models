import cv2 
from ultralytics import YOLO
from ultralytics.solutions.solutions import SolutionAnnotator
from ultralytics.utils.plotting import colors 

model = YOLO("yolo11s.pt")
names = model.names

cap = cv2.VideoCapture("Best-Object-Detection-models/Yolo-V11/How to display Text and Circle Annotations/test.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
writer = cv2.VideoWriter("Best-Object-Detection-models/Yolo-V11/How to display Text and Circle Annotations/test_with_annotations.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    annotator = SolutionAnnotator(im0)
    results = model.predict(im0) 
    boxes = results[0].boxes.xyxy.cpu() 
    clss = results[0].boxes.cls.cpu().tolist() 

    for box , cls in zip(boxes, clss):

        # Circle annotation
        annotator.adaptive_label(box, label=names[int(cls)] , color=colors(cls,True), shape="circle")

       # Rectangle annotation
        #annotator.adaptive_label(box, label=names[int(cls)] , color=colors(cls,True), shape="rect")


    writer.write(im0)
    cv2.imshow("image with anotations", im0) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()


                         