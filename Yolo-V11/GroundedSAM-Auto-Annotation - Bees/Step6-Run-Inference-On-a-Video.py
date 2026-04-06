import cv2 
import os 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ultralytics import YOLO

# Weights path 
WEIGHTS_PATH = "d:/temp/model/Bees-model/Bees-small/weights/best.pt"

# Load the model 
model = YOLO(WEIGHTS_PATH)

# Define Path to video file
INPUT_VIDEO_PATH = "Best-Object-Detection-models/Yolo-V11/GroundedSAM-Auto-Annotation - Bees/Videos/Test videos/test_video.mp4" 

# Extract folder and file name from the input video path 
input_folder = os.path.dirname(INPUT_VIDEO_PATH) 
input_filename = os.path.basename(INPUT_VIDEO_PATH)
output_video_path = os.path.join(input_folder , f"annotated_{input_filename}")

# Open the input video file
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

# Get the video properties 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create the output video writer 
out = cv2.VideoWriter(output_video_path , fourcc , fps , (frame_width , frame_height))

# Init matplotlib figure for display the frames
fig , ax = plt.subplots(figsize=(12,8))
frame_display = None 

# Function to update frames in Matplotlib 
def update_frame(_):
    global frame_display 

    # Read the next frame 
    success , frame = cap.read()
    if not success :
        return
    
    # Run the Yolo inference on the frame 
    results = model(frame) 

    # Visualize the results 
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video 
    out.write(annotated_frame)

    # Convert frame from BGR to RGB for Matplotlib 
    annotated_frame = cv2.cvtColor(annotated_frame , cv2.COLOR_BGR2RGB)

    # Display the annotated frame 
    if frame_display is None :
        frame_display = ax.imshow(annotated_frame)
    else :
        frame_display.set_data(annotated_frame)
    


# Configure the Matplotlib plot 
plt.axis("off")

ani = FuncAnimation(fig , update_frame , interval=100 // fps)

# Display the animation
plt.show()

# Release the video capture and writer 
cap.release()
out.release()

print(f"Output video saved to : {output_video_path}")




