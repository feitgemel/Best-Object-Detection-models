import supervision as sv 

IMAGE_DIR_PATH = "C:/Data-sets/Horse-race/Source-Data/All-images"

image_paths = sv.list_files_with_extensions(directory=IMAGE_DIR_PATH, extensions=["png", "jpg"])

print("image count : ", len(image_paths))

SAMPLE_SIZE = 16 
SAMPLE_GRID_SIZE = (4,4)
SAMPLE_PLOT_SIZE = (16, 16)

import cv2 

titles = [
    image_path.stem
    for image_path in image_paths[:SAMPLE_SIZE] ]

images = [
    cv2.imread(str(image_path))
    for image_path in image_paths[:SAMPLE_SIZE] ]

sv.plot_images_grid(images=images , titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)

