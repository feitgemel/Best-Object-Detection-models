import xml.etree.ElementTree as ET
import os

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



# Run the code:

#Step 1 - only 1 file                      
json_file_path = 'C:/Data-sets/Ships detection/training/annotations/GE_1.xml'
yolo_output_path = 'C:/Data-sets/Ships detection/training/labels/GE_1.txt'

# Define class indices based on your dataset
class_indices = {"boat": 0}



# Example usage
lables_folder_path = 'C:/Data-sets/Ships detection/training/labels'
if not os.path.exists(lables_folder_path):
        os.makedirs(lables_folder_path)


convert_xml_to_yolo(json_file_path, yolo_output_path, class_indices)

