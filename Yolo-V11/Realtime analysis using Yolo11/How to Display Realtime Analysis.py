import cv2
from ultralytics import solutions

# open the video file 
cap = cv2.VideoCapture('Best-Object-Detection-models/Yolo-V11/Realtime analysis using Yolo11/people.mp4') 
assert cap.isOpened(), "Error reading video file"

# Get the original Video properties 
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)) 

# Init video Writer 
out = cv2.VideoWriter(
    "Best-Object-Detection-models/Yolo-V11/Realtime analysis using Yolo11/output_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),)

# init Analytics solution 

analytics = solutions.Analytics(
    model="yolo11n.pt",
    analytics_type="line",
    show=False,
    classes=[0] # 0-> a person . You can add more classess 
)

frame_count = 0 

while cap.isOpened(): 
    success , im0 = cap.read() 
    
    if success:
        frame_count += 1

        # Snapshot of the clean frame for the video thumbnail 
        raw_thumbnail = im0.copy() 

        # 1. Process the frame 
        results = analytics.process(im0, frame_count)
        annotated_frame = results.plot_im

        if annotated_frame is not None: 
            # 2. Match reolution if necessary 
            if (annotated_frame.shape[1] != w) or (annotated_frame.shape[0] != h): 
                annotated_frame = cv2.resize(annotated_frame, (w, h))

            # 3. Create the picture in picture effect 
            scale_precent = 0.3 
            thumb_w , thumb_h = int(w * scale_precent), int(h * scale_precent)
            thumbnail = cv2.resize(raw_thumbnail, (thumb_w, thumb_h))


            # 4 Position adjustment 
            margin_right = 20 
            start_y = int( h * 0.35)
            start_x = w - thumb_w - margin_right

            # Overlay the thumbnail
            annotated_frame[start_y:start_y + thumb_h, start_x:start_x + thumb_w] = thumbnail 

            # 5. Write and display 
            out.write(annotated_frame) 
            cv2.imshow("Analysis", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break 
    else: 
        break


# Clean
cap.release()
out.release()
cv2.destroyAllWindows()