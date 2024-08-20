import os

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



# Run eh code:

#Step 1 - only 1 file                      
json_file_path = 'C:/Data-sets/Stanford Dogs Dataset/annotations/Annotation/n02087394-Rhodesian_ridgeback/n02087394_2253'

yolo_output_path = 'C:\Data-sets/Stanford Dogs Dataset/annotations/n02087394_2253.txt'


# Example usage
lables_folder_path = 'C:/Data-sets/Stanford Dogs Dataset/training/labels'
if not os.path.exists(lables_folder_path):
        os.makedirs(lables_folder_path)


convert_xml_to_yolo(json_file_path, yolo_output_path, class_indices)


