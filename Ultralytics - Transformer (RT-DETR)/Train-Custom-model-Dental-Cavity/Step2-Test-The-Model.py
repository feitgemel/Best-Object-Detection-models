from ultralytics import RTDETR
import cv2

# load the saved model
model = RTDETR("D:/Temp/Models/RT-DETR-Cavity/dental-cavity/weights/best.pt")

# Load a test image from the test folder 
imgTest = "D:/Data-Sets-Object-Detection/Dental cavity/test/images/healthy_teeth_49_jpg.rf.b9c610d1e79d202a172ff300f1b785e6.jpg"
imgAnot = "D:/Data-Sets-Object-Detection/Dental cavity/test/labels/healthy_teeth_49_jpg.rf.b9c610d1e79d202a172ff300f1b785e6.txt"

img = cv2.imread(imgTest)
H, W, _ = img.shape

# Run an inference on the image
imgPredict = img.copy() 
threshold = 0.5 
results = model(imgPredict)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(imgPredict, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1) 
        cv2.putText(imgPredict, results.names[int(class_id)].upper(),(int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0), 1)



# The Ground Truth

ImageTruth = img.copy()

with open(imgAnot, "r") as file:
     lines = file.readlines()

annotations=[] 
for line in lines:
    values = line.split()
    label = values[0]
    x , y , w, h, = map(float, values[1:])
    annotations.append((label, x, y, w, h))

for annotation in annotations:
    label, x, y, w, h = annotation
    label = results.names[int(label)].upper() # Get the label name 

    # Convert Yolo coordinates to pixel coordinates
    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)

    # Draw the bounding box and label on the image
    cv2.rectangle(ImageTruth, (x1, y1), (x2, y2), (0,255,0),1 )

    # Display the label 
    cv2.putText(ImageTruth, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1 )

cv2.imwrite(r"Best-Object-Detection-models\Ultralytics - Transformer (RT-DETR)/Train-Custom-model-Dental-Cavity/GroundTruth.png",ImageTruth)
cv2.imwrite(r"Best-Object-Detection-models\Ultralytics - Transformer (RT-DETR)/Train-Custom-model-Dental-Cavity/Predicted.png",imgPredict)
cv2.imshow("Image Truth", ImageTruth)
cv2.imshow("Image Predict", imgPredict)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


