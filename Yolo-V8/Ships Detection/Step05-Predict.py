from ultralytics import YOLO
import cv2 
import os 

imgTest = "Best-Object-Detection-models/Yolo-V8/Ships Detection/boats-test.jpg"

img = cv2.imread(imgTest)
H , W , _ = img.shape 

# Predict 

imgpredict = img.copy() 

model_path = os.path.join("C:/Data-sets/Ships detection/My-model","weights","best.pt")

# load the model 
model = YOLO(model_path)

threshold = 0.2 # lets start will low value since we have many types of different ships

results = model(imgpredict)[0]


for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score , class_id = result # get the coorinates of each predict boat + score + class id 

    if score > threshold :
        cv2.rectangle(imgpredict, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)

        cv2.putText(imgpredict, results.names[int(class_id)].upper(), (int(x1), int(y1-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

cv2.imwrite("c:/temp/predict.png", imgpredict)       
cv2.imshow("imgpredict", imgpredict)
cv2.imshow("img", img)

cv2.waitKey(0)