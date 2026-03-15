from ultralytics import YOLO

# Load the engine model 
model = YOLO("yolov8l.engine", task="detect")

# Video file : Football.mp4 
# Send me an email to feitgemel@gmail.com , and I will send you a link for the video 

# Save=True -> We will save the output 
result = model.predict("d:/temp/football.mp4", save=True) 