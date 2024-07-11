from ultralytics import YOLO
import cv2 
import os 

imgTest = "C:/Data-sets/teeth2/test/images/IMG_5623_JPG.rf.0c498b907e44fe1bdba6ee17166bee3e.jpg"
imgAnot = "C:/Data-sets/teeth2/test/labels/IMG_5623_JPG.rf.0c498b907e44fe1bdba6ee17166bee3e.txt"

img = cv2.imread(imgTest)
H , W, _ = img.shape

# predict 

imgPredict = img.copy()

model_path = os.path.join("C:/Data-sets/teeth2/My-Teeth-Model","weights","best.pt")

#load the model
model = YOLO(model_path)
threshold = 0.5 

results = model(imgPredict)
results = results[0]

for result in results.boxes.data.tolist():
    x1 , y1 , x2, y2 , score , class_id = result 

    if score > threshold:
        cv2.rectangle(imgPredict, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1  )
        cv2.putText(imgPredict, results.names[int(class_id)].upper() , (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

#print(results)

# Ground truth 

imgTrue = img.copy()

with open(imgAnot , 'r') as file :
    lines = file.readlines()

annotations = []
for line in lines:
    values = line.split()
    label = values[0]
    x, y, w, h = map(float, values[1:])
    annotations.append((label, x, y, w, h))

for annotation in annotations:
    label , x, y, w, h = annotation

    label = results.names[int(label)].upper() # get the label name (reuse the lables out of the model)

    # Convert YOLO coordinates to pixel coordinates 
    x1 = int ( (x - w / 2) * W)
    y1 = int ( (y - h / 2) * H)
    x2 = int ( (x + w / 2) * W)
    y2 = int ( (y + h / 2) * H)


    # Draw a bounding box 
    cv2.rectangle(imgTrue , (x1, y1), (x2, y2) , (0,255,0), 1)

    # display the label name
    cv2.putText(imgTrue , label, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.5 , (0,255,0), 1, cv2.LINE_AA)


cv2.imwrite("c:/temp/imgTrue.png",imgTrue )
cv2.imwrite("c:/temp/imgPredict.png",imgPredict )

cv2.imshow("Img Predict", imgPredict)
cv2.imshow("Img True", imgTrue)
cv2.imshow("Original ", img)
cv2.waitKey(0)




