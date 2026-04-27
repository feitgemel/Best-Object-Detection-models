import os
import cv2
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

# Paths
images_folder = "D:/Data-Sets-Object-Detection/military/dataset/valid/images"
labels_folder = "D:/Data-Sets-Object-Detection/military/dataset/valid/labels"
model_path = os.path.join("D:/Temp/Models/Military Aircraft Detection","My-Model-224","weights","best.pt")

# Load the YOLO model
model = YOLO(model_path)
threshold = 0.5

# Get all images and labels
images = [img for img in os.listdir(images_folder) if img.endswith(('.jpg', '.png'))]
labels = {os.path.splitext(label)[0]: os.path.join(labels_folder, label) for label in os.listdir(labels_folder)}

# Randomly select 10 images
selected_images = random.sample(images, min(10, len(images)))

# Iterate through the selected images
for img_name in selected_images:
    img_path = os.path.join(images_folder, img_name)
    label_path = labels.get(os.path.splitext(img_name)[0])

    if not label_path:
        print(f"Label file for {img_name} not found, skipping.")
        continue

    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for plt
    H, W, _ = img.shape

    # Copy for predictions and ground truth
    img_predict = img.copy()
    img_truth = img.copy()

    # Predict
    results = model(img_predict)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(img_predict, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
            predicted_label = results.names[int(class_id)].upper()
            cv2.putText(img_predict, predicted_label, (int(x1), int(y1 - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    # Ground truth
    with open(label_path, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        values = line.split()
        label = values[0]
        x, y, w, h = map(float, values[1:])
        annotations.append((label, x, y, w, h))

    for annotation in annotations:
        label, x, y, w, h = annotation
        label_name = results.names[int(label)].upper()

        # Convert YOLO coordinates to pixel coordinates
        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)
        x2 = int((x + w / 2) * W)
        y2 = int((y + h / 2) * H)

        # Draw bounding box and label
        cv2.rectangle(img_truth, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(img_truth, label_name, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    # Add space between images
    spacer = np.ones((H, 50, 3), dtype=np.uint8) * 255
    combined_img = np.hstack((img_truth, spacer, img_predict))

    # Display the combined image using matplotlib
    plt.figure(figsize=(15, 8))  # Increased figure size
    plt.imshow(combined_img)
    plt.title(f"Ground Truth (Left) and Predicted (Right) - {img_name}")
    plt.axis('off')
    plt.show()
