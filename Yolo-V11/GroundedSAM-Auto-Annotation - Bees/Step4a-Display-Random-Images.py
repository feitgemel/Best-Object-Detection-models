import supervision as sv 
from pathlib import Path
import random 

ANNOTATIONS_DIRECTORY_PATH = "D:/Data-Sets-Object-Detection/Bees/dataset/train/labels"
IMAGES_DIRECTORY_PATH = "D:/Data-Sets-Object-Detection/Bees/dataset/train/images"
DATA_YAML_PATH = "D:/Data-Sets-Object-Detection/Bees/dataset/data.yaml"

# Display images 

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH)

print(len(dataset))



SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4,4)
SAMPLE_PLOT_SIZE = (16,10)

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# Shuffle dataset and select random images
dataset_indices = list(range(len(dataset)))
random.shuffle(dataset_indices)


images = [] 
images_names = []

for i in dataset_indices[:SAMPLE_SIZE] :

    image_path , image , annotation = dataset[i]
    
    annotated_image = image.copy() 
    annotated_image = mask_annotator.annotate(
        scene=annotated_image,
        detections=annotation)
    
    annotated_image = box_annotator.annotate(
        scene=annotated_image,
        detections=annotation)
    
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=annotation)
    
    images_names.append(Path(image_path).name)
    images.append(annotated_image)


sv.plot_images_grid(
    images=images,
    titles=images_names,
    grid_size=SAMPLE_GRID_SIZE,
    size=SAMPLE_PLOT_SIZE)
    