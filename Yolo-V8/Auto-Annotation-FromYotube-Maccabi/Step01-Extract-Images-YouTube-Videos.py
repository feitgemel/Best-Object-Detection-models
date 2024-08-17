import cv2 
import os 
from vidgear.gears import CamGear 

train_URLs = ['https://youtu.be/32bsPfx1kmY?si=JzcaFYrmkp9ubB3O',
            'https://youtu.be/QxzdEivMbvE?si=JuJMv-dzVSvkHouG']

numerator = 0
output_path_images = "c:/data-sets/Mac-Real/images"

if not os.path.exists(output_path_images):
    os.makedirs(output_path_images)

for url in train_URLs:
    print(url)

    # Path to the Video dile
    stream = CamGear(source=url , stream_mode=True, logging=True ).start()

    # read and display the video frame by frame
    while True :

        frame = stream.read()
        print(numerator)
        numerator = numerator + 1

        if frame is None:
            break 

        image_output_path = output_path_images + "/" + "images" + str(numerator) + ".png"
        resized = cv2.resize(frame , (640,640) , interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_output_path, resized)

        # Display the frame 
        cv2.putText(frame , "imag no. " + str(numerator), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 4)

        cv2.imshow("img", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break 

cv2.destroyAllWindows()
stream.stop()


