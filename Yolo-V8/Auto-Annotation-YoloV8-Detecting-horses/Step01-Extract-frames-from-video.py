import supervision as sv 
from tqdm.notebook import tqdm 

VIDEO_DIR_PATH = "C:/Data-sets/Horse-race/Source-Data/videos"
IMAGE_DIR_PATH = "C:/Data-sets/Horse-race/Source-Data/All-images"

FRAME_STRIDE=10 

video_paths = sv.list_files_with_extensions(directory=VIDEO_DIR_PATH, extensions=["mov", "mp4"])

print(video_paths)

image_numerator = 0 

for video_path in video_paths:
    print("Extracted video name : " + str(video_path))

    video_name = video_path.stem 
    image_name_pattern = video_name + "-{:05d}.png"

    with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink :
        for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE):

            image_numerator = image_numerator + 1
            print("Extract image no. " + str(image_numerator))
            sink.save_image(image=image)





