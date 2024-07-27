from ultralytics import YOLO 
import cv2 
import yaml 

img = "C:/Data-sets/Playing Cards/train/images/001198293_jpg.rf.411db15ce8a9a42a2d51a1885f7592d2.jpg"
imgAnot = "C:/Data-sets/Playing Cards/train/labels/001198293_jpg.rf.411db15ce8a9a42a2d51a1885f7592d2.txt"


# Get label names 
data_yaml_file = "C:/Data-sets/Playing Cards/data.yaml"

with open(data_yaml_file , 'r') as file:
    data = yaml.safe_load(file)

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
    annotations.append((label , x, y, w, h))

print(annotations)

for annotation in annotations:
    label , x, y, w, h = annotation
    label_name = label_names[int(label)]


    # convert Yolo coordinates to pixel coordinates

    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)

    # Draw bounding box 
    cv2 .rectangle(img , (x1, y1), (x2, y2), (200,200,0), 1)

    # display the label name
    cv2.putText(img , label_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
