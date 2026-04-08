import os
import random
import matplotlib.pyplot as plt
import cv2

label_names = ["an athletic person running race track",
"Athletic runner",
"Olympic athlete",

]

def get_annoations(original_img, label_file):
    with open(label_file, 'r') as file:
        lines = file.readlines()
    
    annotations = []
    for line in lines:
        values = line.split()
        label = values[0]
        x, y, w, h = map(float, values[1:])
        annotations.append((label, x, y, w, h))
    return annotations

def put_annoations_in_image(image, annotations):
    H, W, _ = image.shape
    for annotation in annotations:
        label, x, y, w, h = annotation
        label_name = label_names[int(label)]
        
        # Convert YOLO coordinates to pixel coordinates
        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)
        x2 = int((x + w / 2) * W)
        y2 = int((y + h / 2) * H)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 0), 1)
        
        # Display label
        cv2.putText(image, label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2, cv2.LINE_AA)

    return image

def display_random_images(folder_path, num_images, label_folder):
    # Get list of all image filenames in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Randomly select num_images filenames
    selected_images = random.sample(image_files, num_images)

    # Create a subplot grid for 8 images (2 rows, 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Randomly Selected Images with Annotations')

    # Iterate through the selected images and display them
    for i, image_file in enumerate(selected_images):
        row = i // 4
        col = i % 4
        
        img = cv2.imread(os.path.join(folder_path, image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for matplotlib
        
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_file_path = os.path.join(label_folder, label_file)
        
        # Read annotations
        annotations_Yolo_format = get_annoations(img, label_file_path)
        
        # Put bounding boxes
        image_with_annotations = put_annoations_in_image(img, annotations_Yolo_format)
        
        # Display the image in the subplot
        axes[row, col].imshow(image_with_annotations)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Image {i + 1}")

    plt.tight_layout()
    plt.show()

# Replace 'folder_path' with the path to your folder containing images
images_path = 'D:/Data-Sets-Object-Detection/Athletic-Runner/Dataset/train/images'
label_folder = 'D:/Data-Sets-Object-Detection/Athletic-Runner/Dataset/train/labels'

num_images = 8  # Adjusted to display 8 images
display_random_images(images_path, num_images, label_folder)
