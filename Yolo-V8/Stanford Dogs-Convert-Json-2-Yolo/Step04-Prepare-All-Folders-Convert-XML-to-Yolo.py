import os
import shutil

directory = "C:/Data-sets/Stanford Dogs Dataset/annotations/Annotation"

folder_names = []
for folder in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, folder)):
        folder_names.append(folder)

class_indices = {}
for i, folder_name in enumerate(folder_names):
    name_after_dash = "-".join(folder_name.split("-")[1:]).strip()  # Extract the name after the "-" sign and remove leading/trailing spaces
    class_indices[name_after_dash] = i

print("class_indices =", class_indices)

# ====================================================================================

import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_path, yolo_path, class_indices):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width = int(root.find(".//size/width").text)
    image_height = int(root.find(".//size/height").text)

    with open(yolo_path, 'w') as yolo_file:
        for obj in root.findall(".//object"):
            class_name = obj.find('name').text
            class_index = class_indices[class_name]

            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            # Convert bounding box coordinates to YOLO format
            x_center = (xmin + xmax) / (2.0 * image_width)
            y_center = (ymin + ymax) / (2.0 * image_height)
            box_width = (xmax - xmin) / image_width
            box_height = (ymax - ymin) / image_height

            yolo_line = f"{class_index} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
            yolo_file.write(yolo_line)



# Run the code for all folders:
# =============================            


# create destination folders
            
output_train_images_folder = "C:/Data-sets/Stanford Dogs Dataset/dataset/train/images"
if not os.path.exists(output_train_images_folder):
        os.makedirs(output_train_images_folder)

output_valid_images_folder = "C:/Data-sets/Stanford Dogs Dataset/dataset/valid/images"
if not os.path.exists(output_valid_images_folder):
        os.makedirs(output_valid_images_folder)

output_train_lables_folder = "C:/Data-sets/Stanford Dogs Dataset/dataset/train/labels"
if not os.path.exists(output_train_lables_folder):
        os.makedirs(output_train_lables_folder)

output_valid_lables_folder = "C:/Data-sets/Stanford Dogs Dataset/dataset/valid/labels"
if not os.path.exists(output_valid_lables_folder):
        os.makedirs(output_valid_lables_folder)



# list of original annotations folder
#print(folder_names)

# the output folder for the Yolo lables txt files
source_images_folder = "C:/Data-sets/Stanford Dogs Dataset/images/Images"

split_numerator=0

# copy all the images with 90% train and 10% valid
# =================================================
images_folder_names = os.listdir(source_images_folder)
print(images_folder_names)

for folder in images_folder_names:
     list_of_images = os.listdir( os.path.join(source_images_folder, folder) )
     
     for image in list_of_images:
          
          residue = split_numerator % 10 # every 10 image and label will be valid dataset

          image_full_path = os.path.join(source_images_folder, folder,image )

          if residue == 0 :
                image_full_path_dsitination = os.path.join(output_valid_images_folder, image ) 
          else:
                image_full_path_dsitination = os.path.join(output_train_images_folder, image ) 

          #print("Image file name : " + image_full_path)
          shutil.copyfile(image_full_path, image_full_path_dsitination)



          # generate lables in Yolo format
          # =================================       

          file_name_without_extension = os.path.splitext(image)[0] # get rid of the .jpg
          full_file_path = os.path.join(directory, folder, file_name_without_extension) 
          #print("Lebel : "+full_file_path)

          if residue == 0 :
                yolo_file_path = os.path.join(output_valid_lables_folder, file_name_without_extension+".txt") 
     
          else:
                yolo_file_path = os.path.join(output_train_lables_folder, file_name_without_extension+".txt") 

          convert_xml_to_yolo(full_file_path, yolo_file_path, class_indices)

          split_numerator = split_numerator + 1  
          print("File no. " + str(split_numerator))
        
          

    