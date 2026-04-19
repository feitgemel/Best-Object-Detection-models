from ultralytics import YOLOWorld
import cv2 
import os 

imgTest = "D:/Data-Sets-Object-Detection/trash_ICRA19/YOLO_dataset/test/images/obj0319_frame0000035.jpg"
imgAnot = "D:/Data-Sets-Object-Detection/trash_ICRA19/YOLO_dataset/test/labels/obj0319_frame0000035.txt"

img = cv2.imread(imgTest)
H,W,_ = img.shape 

# Predict
#########

imgPredict = img.copy() 
model_path = os.path.join("D:/Temp/Models/Underwater-Trash", "My-YoloWorld-Model-S","weights", 'best.pt')

# Load the model 
model = YOLOWorld(model_path)
threshold = 0.5 

results = model(imgPredict)[0]

for result in results.boxes.data.tolist() :
    x1, y1, x2, y2, score , class_id = result 

    if score > threshold:
        cv2.rectangle(imgPredict, (int(x1), int(y1)), (int(x2) , int(y2)) ,(0,255,0) ,1)
        cv2.putText(imgPredict, results.names[int(class_id)].upper() , (int(x1),int(y1+30)) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)



# Ground True
#############

import yaml

data_yaml_file = "Best-Object-Detection-models/YOLO-World/Method2-Build-Model-Based-on-Your-custom-dataset/data.yaml"

with open(data_yaml_file,'r') as file:
    data = yaml.safe_load(file)

label_names = data['names']
print(label_names)

imgTrue = img.copy() 
H, W, _ = imgTrue.shape

with open(imgAnot , 'r') as file :
    lines = file.readlines() 


annotations = [] 
for line in lines :
    values = line.split()
    label = values[0]
    x, y, w, h = map(float, values[1:])
    annotations.append((label, x, y, w, h))


for annotation in annotations :
    label, x, y, w, h = annotation
    label_name = label_names[int(label)]

    # Convert Yolo coordinates into pixel coordinates 
    x1 = int((x - w/2) * W)
    y1 = int((y - h/2) * H)
    x2 = int((x + w/2) * W)
    y2 = int((y + h/2) * H) 
             
    # Draw bounding box 
    cv2.rectangle(imgTrue , (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Display label 
    cv2.putText(imgTrue, label_name, (x1, y1 + 30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)



cv2.imshow("Predict", imgPredict)
cv2.imshow("imgTrue", imgTrue)


cv2.waitKey(0)
cv2.destroyAllWindows()
