#Git hub : https://github.com/obss/sahi

import cv2 
from sahi import AutoDetectionModel 
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8m_model
from ultralytics import YOLO 

img_path = "Best-Object-Detection-models/Yolo-V8/yolov8-sahi-Detect-Objects/cars.jpg"
model_path = "yolov8m.pt"

yolov8_model_path = f'models/{model_path}'

# Download the model
download_yolov8m_model(yolov8_model_path)

detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                     model_path=yolov8_model_path,
                                                     confidence_threshold=0.6,
                                                     device='cpu')


# load the image and reduce the size 

img_original = cv2.imread(img_path)
scale_precent = 30 
w = int(img_original.shape[1] * scale_precent / 100)
h = int(img_original.shape[0] * scale_precent / 100)

dim = (w, h )

img_original = cv2.resize(img_original, dim , interpolation = cv2.INTER_AREA)


# YoloV8 - Simple predict

img2 = img_original.copy()
model = YOLO(model_path)
results = model(img2, conf = 0.6)

annotated_frame = results[0].plot()


# Yolov8 + Sahi predict 
img = img_original.copy() 

results = get_sliced_prediction(img , 
                                detection_model, 
                                slice_height=512, 
                                slice_width=512 , 
                                overlap_height_ratio=0.2 , 
                                overlap_width_ratio=0.2)

object_predicition_list = results.object_prediction_list

boxes_list = []
clss_list = [] 

for ind , _ in enumerate(object_predicition_list):

    boxes = object_predicition_list[ind].bbox.minx, object_predicition_list[ind].bbox.miny , object_predicition_list[ind].bbox.maxx , object_predicition_list[ind].bbox.maxy 

    clss = object_predicition_list[ind].category.name 

    boxes_list.append(boxes)
    clss_list.append(clss)


print (clss_list)


for box , cls in zip(boxes_list, clss_list) :
    x1 , y1 , x2, y2 = box 
    cv2.rectangle(img , (int(x1), int(y1)), (int(x2), int(y2)), (56,56,255), 2)

    label = str(cls)
    t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]

    cv2.rectangle(img , (int(x1), int(y1)-t_size[1]-3), (int(x1)+t_size[0], int(y1)+3), (56,56,255) , -1)

    cv2.putText(img , 
                label, 
                (int(x1), int(y1) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7 ,
                [255,255,255], 
                thickness = 2,
                lineType = cv2.LINE_AA)


cv2.imshow("Yolo V8 sAHI ", img)
cv2.imshow("Yolo V8 Simple predicition", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()