from ultralytics import YOLOWorld

# Init the Yolo-World model

model = YOLOWorld('yolov8s-world.pt')

# Step1 - Execute inference on a specific image 
#img_path = "Best-Object-Detection-models/YOLO-World/Method1-Build-custom-Object-detection-model/airplane.jpg" 
img_path = "Best-Object-Detection-models/YOLO-World/Method1-Build-custom-Object-detection-model/train.jpg" 
results = model.predict(img_path, device='cuda') 

# show results

for result in results:
    print(result.boxes.data)

results[0].show()


# Step2 - Set prompts for specific tasks

# Define out classes and save a new model 

model2 = YOLOWorld('yolov8s-world.pt')
model2.set_classes(["cell phone", "bag", "Yellow shoes"])
results2 = model2.predict(img_path, device='cuda', conf=0.85) 
model2.save("d:/temp/models/my-yolov8s-world.pt")

results2[0].show()


