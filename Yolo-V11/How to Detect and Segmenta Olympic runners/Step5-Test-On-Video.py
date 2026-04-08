import cv2 
from ultralytics import YOLO
import os 
import numpy as np 
from segment_anything import sam_model_registry, SamPredictor

# Video source path 
video_path = "Best-Object-Detection-models/Yolo-V11/How to Detect and Segmenta Olympic runners/race.mov"

# Path to the saved model (YoloV11 weights)
model_path = os.path.join("D:/Temp/Models/Athletic-Runner/checkpoits", "Athletic-Runner-nano", "weights", "best.pt")

# Path to the SAM weights 
path_for_sam_model = "d:/temp/models/sam_vit_h_4b8939.pth"

import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "default"

# Init the model 
model = YOLO(model_path)
detection_threshold = 0.7 

# Load the SAM segmentation model 
sam = sam_model_registry[model_type](checkpoint=path_for_sam_model)
sam.to(device=device)
predictor = SamPredictor(sam)


# Video procession loop 
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    # Add the SAM predictor 
    predictor.set_image(frame)


    # Run the Yolo detection 
    results = model(frame)[0]
    classes = results.names 

    # Create a copy of the fram for the final visualization
    display_frame = frame.copy()

    # Iterate all the detections found by Yolo

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Check if the detected object is our target class and meets the threshold 
        # class 1 is "Athletic runner"

        if int(class_id) == 1 and score > detection_threshold:

            # SAM Segmentation step 
            # Convert Yolo coordinates to a Numpy arrays for SAM
            # Format [x_min , y_min , x_max , y_max ]
            input_box = np.array([x1, y1, x2, y2])

            # Use the bounding box as a 'prompt' for SAM
            # This will be more accurte 
            masks , scores , _ = predictor.predict(
                box=input_box,
                multimask_output=False,
            ) 

            # Get the binary mask (A boolean array)
            mask = masks[0]


            # Visual the result 
            
            # 1. Create a random of fixed color for the mask
            # We will use here a green semi transparent overlay 
            mask_overlay = np.zeros_like(display_frame, dtype=np.uint8)
            mask_overlay[mask] = [0,255,0] # Green color 

            # 2. Blend the mask with the frame (Alpha blending)
            # 0.6 is the original frame weight , 0,4 is the mask weight 
            cv2.addWeighted(display_frame, 1.0, mask_overlay, 0.4, 0, display_frame)


            # 3. Draw a bounding box and label 
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            label_text = f"{classes[int(class_id)]}: {score:.2f}"
            cv2.putText(display_frame, label_text, (int(x1), int(y1 - 10 )), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



    # Display the result frame 
    cv2.imshow("Runner detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











































