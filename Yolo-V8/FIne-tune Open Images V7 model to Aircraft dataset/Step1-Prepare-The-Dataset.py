import os
import shutil
from tqdm import tqdm

# Define the folder paths
main_folder = "D:/Data-Sets-Object-Detection/military"
images_folder = os.path.join(main_folder, "images")
labels_folder = os.path.join(main_folder, "labels")
train_images_folder = os.path.join(images_folder, "aircraft_train")
val_images_folder = os.path.join(images_folder, "aircraft_val")
train_labels_folder = os.path.join(labels_folder, "aircraft_train")
val_labels_folder = os.path.join(labels_folder, "aircraft_val")
dataset_folder = os.path.join(main_folder, "dataset")

# Create the "dataset" folder structure
train_dataset_folder = os.path.join(dataset_folder, "train")
valid_dataset_folder = os.path.join(dataset_folder, "valid")

folders_to_create = [
    os.path.join(train_dataset_folder, "images"),
    os.path.join(train_dataset_folder, "labels"),
    os.path.join(valid_dataset_folder, "images"),
    os.path.join(valid_dataset_folder, "labels"),
]

for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)

# Function to copy files
def copy_files(source_images_folder, target_images_folder, source_labels_folder, target_labels_folder):
    image_files = [f for f in os.listdir(source_images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]  # Adjust extensions as needed
    for image_file in tqdm(image_files, desc=f"Copying files from {source_images_folder}"):
        # Copy image
        src_image_path = os.path.join(source_images_folder, image_file)
        dst_image_path = os.path.join(target_images_folder, image_file)
        shutil.copy2(src_image_path, dst_image_path)

        # Copy corresponding label file if it exists
        label_file = os.path.splitext(image_file)[0] + ".txt"
        src_label_path = os.path.join(source_labels_folder, label_file)
        if os.path.exists(src_label_path):
            dst_label_path = os.path.join(target_labels_folder, label_file)
            shutil.copy2(src_label_path, dst_label_path)

# Copy train files
copy_files(train_images_folder, os.path.join(train_dataset_folder, "images"), train_labels_folder, os.path.join(train_dataset_folder, "labels"))

# Copy valid files
copy_files(val_images_folder, os.path.join(valid_dataset_folder, "images"), val_labels_folder, os.path.join(valid_dataset_folder, "labels"))

print("Dataset structure created and files copied successfully!")

