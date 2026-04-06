import os 

VIDEO_DIR_PATH = "Best-Object-Detection-models/Yolo-V11/GroundedSAM-Auto-Annotation - Bees/Videos/Train videos" 
IMAGE_DIR_PATH = "D:/Data-Sets-Object-Detection/Bees/images"
FRAME_STRIDE = 5 

# We will save each 5 frames ot our image folder 

import supervision as sv 
from tqdm import tqdm

video_paths = sv.list_files_with_extensions(
    directory=VIDEO_DIR_PATH,
    extensions=["mov","mp4"] )

print(video_paths)

for video_path in tqdm(video_paths) :
    video_name = video_path.stem
    image_name_pattern = video_name + "-{:05d}.png"

    with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink :
        for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE) :
            sink.save_image(image=image)

print("Done")


# Check the extracted images folder 
image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["png", "jpg", "jpeg"] )

print("Image Count : ", len(image_paths))

