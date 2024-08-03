import xml.etree.ElementTree as ET
import os

def convert_xml_folder_to_yolo(xml_folder, yolo_folder, class_indices):
    # Create YOLO folder if it doesn't exist
    if not os.path.exists(yolo_folder):
        os.makedirs(yolo_folder)

    # Process each XML file in the folder
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)
            yolo_file = os.path.join(yolo_folder, os.path.splitext(xml_file)[0] + ".txt")

            convert_xml_to_yolo(xml_path, yolo_file, class_indices)

def convert_xml_to_yolo(xml_path, yolo_path, class_indices):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width = int(root.find(".//size/width").text)
    image_height = int(root.find(".//size/height").text)

    with open(yolo_path, 'w') as yolo_file:
        for obj in root.findall(".//object"):
            class_name = obj.find('name').text
            class_index = class_indices.get(class_name, -1)  # Default to -1 if class not found

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



# Define class indices based on your dataset
class_indices = {"boat": 0}

# convert all xml in train folder
xml_train_folder_path = 'C:/Data-sets/Ships detection/training/annotations'
yolo_train_output_folder = 'C:/Data-sets/Ships detection/training/labels'

convert_xml_folder_to_yolo(xml_train_folder_path, yolo_train_output_folder, class_indices)

# convert all xml in test folder
xml_test_folder_path = 'C:/Data-sets/Ships detection/test/annotations'
yolo_test_output_folder = 'C:/Data-sets/Ships detection/test/labels'

convert_xml_folder_to_yolo(xml_test_folder_path, yolo_test_output_folder, class_indices)
