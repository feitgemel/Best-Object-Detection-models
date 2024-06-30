import torch 
import numpy as np 
import cv2 

imgPath = "haverim.jpg"
img = cv2.imread(imgPath)

# load the YoloV5 model from the Pytorch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

imgForModel = [img]

# lets detect a specific frame and score it using the YoloV5 model
# The output would be lables and bounding boxes , that detected by the model

print ("Results : ")
results = model(imgForModel)
print(results)

# This selects all the rows (:) of the first elemnt [0] in the results object 
lables = results.xyxyn[0][: , -1] # all the values in the first column , and the last index 
print(lables)

# Extract the lables class for example :
classes = model.names
print("Classes : ")
print(classes)

class0 = classes[int(lables[0])]
class7 = classes[int(lables[7])]

print(class0)
print(class7)

# extract the coordinates (Bounding boxes for each object)
cords = results.xyxyn[0][: , :-1] # get all the values untill the last coordinate

print("Coordinates : ")
print(cords)


n = len(lables) # number of detected objects
x_shape , y_shape = img.shape[1], img.shape[0]

for i in range(n):
    row = cords[i]

    if row[4] >= 0.2 :
        x1 = int(row[0] * x_shape)
        y1 = int(row[1] * y_shape)
        x2 = int(row[2] * x_shape)
        y2 = int(row[3] * y_shape)
        box_color = (255, 0 , 0 ) # blue

        cv2.rectangle(img , (x1,y1) , (x2,y2) , box_color, 2)
        className = classes[int(lables[i])]
        cv2.putText(img , className, (x1,y1) , cv2.FONT_HERSHEY_COMPLEX, 1, box_color, 2)

#cv2.imwrite("result.png", img)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()