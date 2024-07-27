from ultralytics import YOLO
import cv2 
import os 

imgTest = "C:/Data-sets/Playing Cards/test/images/008090758_jpg.rf.caa872ea30359a5f1cf5c6de034b8684.jpg"
imgAnnot = "C:/Data-sets/Playing Cards/test/labels/008090758_jpg.rf.caa872ea30359a5f1cf5c6de034b8684.txt"

img = cv2.imread(imgTest)
H , W , _ = img.shape


# predict 
imgpredict = img.copy()
model_path = os.path.join("C:/Data-sets/Playing Cards","My-Card-Model","weights","best.pt")

# load the model 
model = YOLO(model_path)
threshold = 0.5 

results = model(imgpredict)[0]

print(results)

for result in results.boxes.data.tolist():
    x1 , y1 , x2 , y2 , score, class_id = result 

    if score > threshold:
        cv2.rectangle(imgpredict, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
        cv2.putText(imgpredict, results.names[int(class_id)].upper(), (int(x1), int(y1-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        
# Ground truth

imgTruth = img.copy()
with open(imgAnnot, 'r') as file:
    lines = file.readlines()


annotations = [] 
for line in lines:
    values = line.split()
    label = values[0]

    x, y, w, h = map(float, values[1:])
    annotations.append((label , x, y, w, h))

print(annotations)

for annotation in annotations:
    label , x, y, w, h = annotation
    label_name = results.names[int(label)].upper()


    # convert Yolo coordinates to pixel coordinates

    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)

    # Draw bounding box 
    cv2 .rectangle(imgTruth , (x1, y1), (x2, y2), (200,200,0), 1)

    # display the label name
    cv2.putText(imgTruth , label_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 2)

cv2.imwrite("c:/temp/imgTruth.png",imgTruth)
cv2.imwrite("c:/temp/imgpredict.png",imgpredict)

cv2.imshow("imgTruth", imgTruth)
cv2.imshow("imgpredict", imgpredict)
cv2.imshow("Original", img)



cv2.waitKey(0)

