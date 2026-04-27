from ultralytics import YOLO
import cv2 
import yaml

img = "D:/Data-Sets-Object-Detection/military/dataset/train/images/0a66768d2d566e9bbbc426354db65277.jpg"
imgAnot =  "D:/Data-Sets-Object-Detection/military/dataset/train/labels/0a66768d2d566e9bbbc426354db65277.txt"

# Get label names
data_yaml_file = "Best-Object-Detection-models/Yolo-V8/FIne-tune Open Images V7 model to Aircraft dataset/data.yaml" 

with open(data_yaml_file, 'r') as file:
    data = yaml.safe_load(file)

print(data)
label_names = data['names']
print(label_names)

img = cv2.imread(img)
H, W, _ = img.shape

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
    label_name = label_names[int(label)]

    # Convert Yolo coordinates to pixel coordinates 
    x1 = int((x - w/2) * W)
    y1 = int((y - h/2) * H)
    x2 = int((x + w/2) * W)
    y2 = int((y + h/2) * H) 
             
    # Draw bounding box 
    cv2.rectangle(img, (x1, y1), (x2, y2), (200,200 , 0 ), 1)

    # Display the label 
    cv2.putText(img, label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (200,200,0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

