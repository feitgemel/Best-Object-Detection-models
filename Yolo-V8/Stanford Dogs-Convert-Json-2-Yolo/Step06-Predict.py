from ultralytics import YOLO
import cv2 
import os 

imgPath = "Best-Object-Detection-models/Yolo-V8/Stanford Dogs-Convert-Json-2-Yolo/doberman.jpg"
#imgPath = "Best-Object-Detection-models/Yolo-V8/Stanford Dogs-Convert-Json-2-Yolo/Dori.jpg"


img = cv2.imread(imgPath) 

H, W, _ = img.shape

# Predict 

# load the model 
model_path = os.path.join("C:/Data-sets/Stanford Dogs Dataset/dataset","Nano-Model", "weights", "best.pt")
model = YOLO(model_path)

threshold = 0.3 

results = model(img)[0]

for result in results.boxes.data.tolist():
    x1, y1 , x2, y2 , score , class_id = result 

    if score > threshold :
        cv2.rectangle(img , (int(x1), int(y1)), (int(x2), int(y2)) , (0,0,0) , 3)
        cv2.putText(img , results.names[int(class_id)].upper() , (int(x1), int(y1-10)) ,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0) , 1 , cv2.LINE_AA)

cv2.imshow("img", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
