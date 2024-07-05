
# pip install grad-cam 

import torch 
import cv2 
import numpy as np 
import torchvision.transforms as trasforms
from pytorch_grad_cam import EigenCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image

COLORS = np.random.uniform(0, 255 , size=(80,3))

def parse_detections(results):
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict() 
    boxes , colors , names = [] , [], [] 

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.4:
            continue

        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])

        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        boxes.append((xmin , ymin, xmax, ymax))
        colors.append(color)
        names.append(name)

    
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names) :
        xmin , ymin , xmax , ymax = box 

        cv2.rectangle(img , (xmin , ymin), (xmax, ymax), color , 2)
        cv2.putText(img, name , (xmin , ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , color, 2, lineType=cv2.LINE_AA)


    return img 


# run the image inference detection

imgPath = "images/cows.jpg"
img = cv2.imread(imgPath)
img = cv2.resize(img , (640, 640))
rgb_img = img.copy()
img = np.float32(img) / 255
transform = trasforms.ToTensor()
tensor = transform(img).unsqueeze(0)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#print(model.eval())
#print(model.cpu())

target_layers = [model.model.model.model[-2]]

results = model([rgb_img])

#print(results)

boxes , colors , names = parse_detections(results)

print(boxes)
print(names)

detections = draw_detections(boxes, colors, names, rgb_img.copy())

cv2.imshow("Detections", detections)

# let's create out CAM model and run it over our image

model.to('cuda')

cam = EigenCAM(model, target_layers)
grayscale_cam = cam(tensor)[0, : , : ]
cam_image = show_cam_on_image(img , grayscale_cam, use_rgb=True)

cv2.imshow("cam image", cam_image)

cv2.waitKey(0)









