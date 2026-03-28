from ultralytics import RTDETR 
import cv2 

# Load the pre-trained RT-DETR model 
model = RTDETR("rtdetr-l.pt") # "rtdetr-x.pt" -> Better acuuracy , but slower

# load the image 
image_path = "Best-Object-Detection-models/Ultralytics - Transformer (RT-DETR)/Simple-Object-Detection/car2.jpg"

# Run inference on the image 

results = model(image_path, show=True, save=True, project="d:/temp", name="car2")

# Display the original image 
cv2.imshow("Original Image", cv2.imread(image_path))
cv2.waitKey(0)
cv2.destroyAllWindows()


# load the original video :
video_path = "Best-Object-Detection-models/Ultralytics - Transformer (RT-DETR)/Simple-Object-Detection/Birds.mp4"

# Run the inference 
results = model(video_path, show=True, save=True)




