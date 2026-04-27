from ultralytics import YOLO
import cv2
import os 

imgTest="Best-Object-Detection-models/Yolo-V8/FIne-tune Open Images V7 model to Aircraft dataset/test.jpg"
imgAnot="Best-Object-Detection-models/Yolo-V8/FIne-tune Open Images V7 model to Aircraft dataset/test.txt"

img = cv2.imread(imgTest)

# Resize image by 30%
img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

H,W,_ = img.shape

# Predict

imgPredict = img.copy()
model_path = os.path.join("D:/Temp/Models/Military Aircraft Detection","My-Model-224","weights","best.pt")

# Load the model 
model = YOLO(model_path)
threshold = 0.5

results = model(imgPredict)[0]
print(results) 

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(imgPredict, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 5)
        cv2.putText(imgPredict, results.names[int(class_id)].upper(), (int(x1), int(y1 - 20 )),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
        

# Display the Ground truth 
imgTruth = img.copy()
with open(imgAnot, 'r') as file:
    lines = file.readlines()


annotations = [] 
for line in lines:
    values = line.split()
    label = values[0]
    x, y, w, h = map(float, values[1:])
    annotations.append((label, x, y, w, h))


for annotation in annotations:
    label, x, y, w, h = annotation

    print(label, x, y, w, h)
    print(results.names)

    labelName = results.names[int(label)].upper() # Get the label name 
    print(labelName)

    # Convert Yolo coordinates to pixel coordinates
    x1 = int((x - w/2) * W)
    y1 = int((y - h/2) * H)
    x2 = int((x + w/2) * W)
    y2 = int((y + h/2) * H) 

    # Draw bounding box 
    cv2.rectangle(imgTruth, (x1, y1), (x2, y2), (0, 255, 0), 5) 
    # Display the label 
    cv2.putText(imgTruth, labelName, (x1, y1 - 20 ) , cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA) 


cv2.imshow("imgTruth", imgTruth)
cv2.imshow("Predict", imgPredict)
cv2.waitKey(0)
cv2.destroyAllWindows()