import cv2 
from ultralytics import YOLO
import os 
from vidgear.gears import CamGear

train_URLs = ['https://www.youtube.com/watch?v=Ox0_uqR_UqA',
            'https://www.youtube.com/watch?v=JUycKVrvZ3c']

numerator = 0 
output_path_images = "D:/Data-Sets-Object-Detection/Athletic-Runner/images"
if not os.path.exists(output_path_images):
    os.makedirs(output_path_images)



for url in train_URLs:
    print(url)

    # Path to the video file 
    stream = CamGear(source=url, stream_mode=True, logging=True).start() # Youtube Url is the input video

    # Read abd display the video frame by frame

    while True:
        
        frame = stream.read()
        print(numerator)
        numerator = numerator + 1

        # Check of the frame is None
        if frame is None:
            break 

        image_output_path = output_path_images + "/" + "image-" + str(numerator) + ".png" 
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_output_path, resized)

        # Display the frame
        cv2.putText(frame, "Image no. " + str(numerator), (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 4, cv2.LINE_AA) 
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
stream.stop()

