import cv2 
import yaml 

imgPath = "D:/Data-Sets-Object-Detection/trash_ICRA19/YOLO_dataset/train/images/obj1672_frame0000134.jpg"
imgAnot = "D:/Data-Sets-Object-Detection/trash_ICRA19/YOLO_dataset/train/labels/obj1672_frame0000134.txt"

# Get label names 

data_yaml_file = "Best-Object-Detection-models/YOLO-World/Method2-Build-Model-Based-on-Your-custom-dataset/data.yaml"

with open(data_yaml_file,'r') as file:
    data = yaml.safe_load(file)

label_names = data['names']
print(label_names)

img = cv2.imread(imgPath)
H, W, _ = img.shape

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
    cv2.rectangle(img , (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Display label 
    cv2.putText(img, label_name, (x1, y1 - 5) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
