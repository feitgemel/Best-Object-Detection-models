from ultralytics import YOLO 
import cv2 
import os 

imgTest = "C:/Data-sets/Bone fracture detection/test/images/all_0_2324_png.rf.64c3541a9e2a4eafb1730138f27ac1e5.jpg"
imgAnot = "C:/Data-sets/Bone fracture detection/test/labels/all_0_2324_png.rf.64c3541a9e2a4eafb1730138f27ac1e5.txt"

img = cv2.imread(imgTest)
H , W, _ = img.shape 


# Predict :

imgPredict = img.copy() 

model_path = os.path.join("C:/Data-sets/Bone fracture detection","My-model","weights","best.pt")

#load the model 
model = YOLO(model_path)

threshold = 0.5 

results = model(imgPredict)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score , class_id = result

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if score > threshold :
        cv2.rectangle(imgPredict, (x1,y1), (x2,y2), (0,255,0), 1)

        class_name = results.names[int(class_id)].upper()

        cv2.putText(imgPredict, class_name, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0), 1, cv2.LINE_AA)


# Ground Truth

imgTruth = img.copy()

with open(imgAnot,'r') as file:
    lines = file.readlines()


annotations = [] 

for line in lines :
    values = line.split()
    label = values[0]
    x, y, w, h = map(float, values[1:])
    annotations.append((label, x, y, w, h))

for annotation in annotations:
    label , x, y, w, h, = annotation

    label = results.names[int(label)].upper()

    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)

    # Draw bounding box 
    cv2.rectangle(imgTruth, (x1,y1), (x2, y2), (0,255,0), 1)

    # Display label
    cv2.putText(imgTruth, label,(x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0), 1)

cv2.imwrite("c:/temp/imgTruth.png", imgTruth)

cv2.imshow("Image Predict", imgPredict)
cv2.imshow("Image Truth", imgTruth)
cv2.imshow("img", img)
cv2.waitKey(0)
