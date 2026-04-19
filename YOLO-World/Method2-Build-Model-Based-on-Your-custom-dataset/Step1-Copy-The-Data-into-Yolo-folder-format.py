import os 
import shutil
from tqdm import tqdm

# Copy function

def copy_files(source , dest_images, dest_labels) :

    os.makedirs(dest_images , exist_ok=True)
    os.makedirs(dest_labels , exist_ok=True)

    for file in tqdm(os.listdir(source)) :
        src_file = os.path.join(source , file)
        if file.endswith((".jpg", ".png" , ".jpeg")): # Copy images 
            shutil.copy(src_file , os.path.join(dest_images , file))
        elif file.endswith(".txt"): # Copy labels 
            shutil.copy(src_file , os.path.join(dest_labels , file))



SOURCE_FOLDER = "D:/Data-Sets-Object-Detection/trash_ICRA19/dataset/"
TARGET_FOLDER = "D:/Data-Sets-Object-Detection/trash_ICRA19/yolo_dataset/"

os.makedirs(TARGET_FOLDER , exist_ok=True)

# Copy train data 
print("Train data copy ......")

copy_files(os.path.join(SOURCE_FOLDER , "train") ,
           os.path.join(TARGET_FOLDER, 'train' , 'images'),
           os.path.join(TARGET_FOLDER, 'train' , 'labels'))


# Copy validation data 
print("Validation data copy ......")

copy_files(os.path.join(SOURCE_FOLDER , "val") ,
           os.path.join(TARGET_FOLDER, 'val' , 'images'),
           os.path.join(TARGET_FOLDER, 'val' , 'labels'))

# Copy test data 

copy_files(os.path.join(SOURCE_FOLDER , "test") ,
           os.path.join(TARGET_FOLDER, 'test' , 'images'),
           os.path.join(TARGET_FOLDER, 'test' , 'labels'))

print("Done")
