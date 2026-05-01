from ultralytics import YOLO
import cv2
import numpy as np

weights_file = "d:/temp/models/african-wildlife-detection/My-Model-yolo26m/weights/best.pt"

model = YOLO(weights_file) 

imgPath1 = "Best-Object-Detection-models/YOLO26/How to detect African Wildlife animals using YoloV2026/Elephant2.jpg"
imgPath2 = "Best-Object-Detection-models/YOLO26/How to detect African Wildlife animals using YoloV2026/Zebra_test.jpg"

results = model.predict([imgPath1,imgPath2]) # Returns a list of results objects 

# Get Classes names 
names_dict = results[0].names
print("Categories : ")
print(names_dict)


# Run the resuls list
for i, result in enumerate(results):
    boxes = result.boxes # Boxes object for bountding box outpus 
    masks = result.masks # Masks object for segmentation masks outputs
    keypoints = result.keypoints # Keypoints object for pose outputs
    probs = result.probs # Probs object for classification outputs
    obb = result.obb # Oriented boxes object for OBB outputs 

    class_ids = result.boxes.cls.cpu().numpy().astype(int) # Convert to Numpy array of integers 
    print("Predicted class Ids:", class_ids)

    # Get Class names 
    class_names = [names_dict[class_id] for class_id in class_ids]
    print("Predicted class names:", class_names)

    # Extract the result image 
    annotated_frame = result.plot()
    cv2.imwrite(f"Best-Object-Detection-models/YOLO26/How to detect African Wildlife animals using YoloV2026/output_image_{i}.jpg", annotated_frame)

    result.show() # Display the screen 




    