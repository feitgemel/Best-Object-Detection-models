# Download files from here :
#https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
# Download MobileNet-SSD v3 : weights and config (both files)

#Download the file and copy it to a folder you like :
# https://github.com/pjreddie/darknet/blob/master/data/coco.names

import cv2 

config_file = "E:/Object-Detection-Models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "E:/Object-Detection-Models/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
coco_lables = "Best-Object-Detection-models/SSD-MobileNet-V3/coco.names"

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []

with open(coco_lables, 'rt') as file :
    classLabels = file.read().rstrip('\n').split('\n')

print(len(classLabels))
print(classLabels)

# Object Detection in an image 

testImagePath = "Best-Object-Detection-models/SSD-MobileNet-V3/man-car.jpg"
img = cv2.imread(testImagePath)


# model configuration
model.setInputSize(320,320)
model.setInputScale( 1.0 / 127.5)
model.setInputMean((127.5 , 127.5, 127.5)) 
model.setInputSwapRB(True)

# Detection
class_ids , confidences , boxes = model.detect(img , confThreshold=0.6)
print(class_ids)

# loop through the detected objects and draw bounding boxes 
for class_id , confidence , box in zip(class_ids, confidences, boxes):
    left , top , width , height = box 
    label = classLabels[class_id - 1] # since the model numerator starts with 1 and not with 0 
    print(label)

    cv2.rectangle(img , (left, top) , (left+width , top+height) , (0,255,0), 2)
    cv2.putText(img , label, (left , top-10 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0) , 2)


cv2.imshow("img", img)
cv2.waitKey(0)


# Object Detection in a Video

cap = cv2.VideoCapture("Best-Object-Detection-models/SSD-MobileNet-V3/video.mp4")
fontScale = 3
font = cv2.FONT_HERSHEY_SIMPLEX

numOfFrames = 0 # just to count and break after about 5 seconds of the video

while True :
    ret , frame = cap.read()
    numOfFrames = numOfFrames + 1
    if numOfFrames >45 :
        break # stop after 45 frames (about 5 seconds)

    scale_percent = 40 # reduce the original size before the model inference
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width , height)
    resized = cv2.resize(frame , dim , interpolation = cv2.INTER_AREA)

    class_ids , confidences , boxes = model.detect(resized , confThreshold=0.6)
    #print(class_ids)

    if (len(class_ids) !=0 ):
        for class_id , confidence , box in zip(class_ids , confidences, boxes):
            left, top , width , height = box 
            class_id = class_id - 1 

            # the 80 classes list start with 1 

            if class_id>0 and class_id< 81 :
                label = classLabels[class_id - 1]
                print(label)
                cv2.rectangle(resized , (left, top) , (left+width , top+height) , (0,255,0), 2)
                cv2.putText(resized , label, (left , top-10 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0) , 2)

    cv2.imshow("resized", resized)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
                



                                               












