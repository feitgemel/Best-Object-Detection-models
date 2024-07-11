from ultralytics import YOLO 
import cv2 
import yaml 

img = "C:/Data-sets/teeth2/train/images/IMG_2135_JPG.rf.b7605f894721641aca14678cbdc8b2f7.jpg"
imgAnotation = "C:/Data-sets/teeth2/train/labels/IMG_2135_JPG.rf.b7605f894721641aca14678cbdc8b2f7.txt"

# get the ymal file
data_yaml_file = "C:/Data-sets/teeth2/data.yaml"

with open(data_yaml_file, 'r') as file :
    data = yaml.safe_load(file)

label_names = data['names']
print(label_names)

img = cv2.imread(img)
H, W, _ = img.shape

with open(imgAnotation, 'r') as file:
    lines = file.readlines()


annotations = []
for line in lines :
    values = line.split()
    label = values[0]
    x, y, w, h = map(float, values[1:])
    annotations.append((label, x, y, w, h))

print(annotations)

for annotation in annotations:
    label , x, y, w, h = annotation
    label_name = label_names[int(label)]

    # convert YOLO coordinates to pixel coordinates 

    x1 = int ( (x - w / 2) * W)
    y1 = int ( (y - h / 2) * H)
    x2 = int ( (x + w / 2) * W)
    y2 = int ( (y + h / 2) * H)


    # Draw a bounding box 
    cv2.rectangle(img , (x1, y1), (x2, y2) , (0,255,0), 1)

    # display the label name
    cv2.putText(img , label_name, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.5 , (0,255,0), 1, cv2.LINE_AA)

cv2.imwrite("c:/temp/1.png", img)
cv2.imshow("img", img)
cv2.waitKey(0)








