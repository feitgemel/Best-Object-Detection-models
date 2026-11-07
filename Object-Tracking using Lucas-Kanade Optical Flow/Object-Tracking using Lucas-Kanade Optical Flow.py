# pip install opencv-python numpy 

import cv2 
import numpy as np

# 1. Open video stream 
test_video_path = "Best-Object-Detection-models/Object-Tracking using Lucas-Kanade Optical Flow/A BMX biker.mp4" 

cap = cv2.VideoCapture(test_video_path)
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the video file.")
    exit() 

# Resize the first frame for faster processing 50%
first_frame = cv2.resize(first_frame, (0, 0), fx=0.5, fy=0.5)

# Initialize VideoWriter to save the output video
output_video_path = "Best-Object-Detection-models/Object-Tracking using Lucas-Kanade Optical Flow/output_tracked_trajectory.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0  # Fallback FPS if video metadata is missing

h_out, w_out = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w_out, h_out))

# 2. Select the region of interest (ROI) for tracking using mouse drag 
roi = cv2.selectROI("Select Target", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Target")

x, y, w, h = roi
print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")
if w == 0 or h == 0:
    print("Error: No ROI selected.")
    out.release()
    cap.release()
    exit()

# 3. Detect initial corner features in the selected ROI
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(prev_gray)
mask[y:y + h, x:x + w] = 255  # Create a mask for the ROI

p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=25, qualityLevel=0.1, minDistance=7) 

# 4. Tracking and trajectory setup 
lk_params = dict(
    winSize=(19, 19),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

trajectory_points = []  # List to store trajectory points
trail_canvas = np.zeros_like(first_frame)  # Canvas to draw the trajectory
smoothed_center = None  # Initialize smoothed center point
alpha = 0.35  # Smoothing factor for exponential moving average

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing 50%
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Calculate optical flow to track the features
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None and st is not None:
        good_points = p1[st == 1]

        if len(good_points) > 0: 
            # Calculate the center of the tracked points
            raw_x = np.median(good_points[:, 0])
            raw_y = np.median(good_points[:, 1])

            # Simple exponential smoothing
            if smoothed_center is None:
                smoothed_center = (raw_x, raw_y)
            else:
                sx = alpha * raw_x + (1 - alpha) * smoothed_center[0]
                sy = alpha * raw_y + (1 - alpha) * smoothed_center[1]
                smoothed_center = (sx, sy)

            # Store position and keep the last 35 points 
            trajectory_points.append((int(smoothed_center[0]), int(smoothed_center[1]))) 
            if len(trajectory_points) > 35:
                trajectory_points.pop(0)

            # Draw tracked feature points (green) 
            for pt in good_points:
                px, py = pt.ravel()
                cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 0), -1)

            # Draw center point (red)
            cv2.circle(frame, (int(smoothed_center[0]), int(smoothed_center[1])), 6, (0, 0, 255), -1)

            # Prepare points for the next frame 
            p0 = good_points.reshape(-1, 1, 2)

    # 6. Fade the trail and draw path lines 
    trail_canvas = cv2.multiply(trail_canvas, 0.92)  # Fade the trail

    if len(trajectory_points) >= 2:
        for i in range(1, len(trajectory_points)):
            cv2.line(trail_canvas, trajectory_points[i - 1], trajectory_points[i], (0, 165, 255), 3)  # Draw the trajectory line (orange)

    # 7. Merge trail with the frame and display the result
    output_frame = cv2.add(frame, trail_canvas)
    
    # Write processed frame to output video file
    out.write(output_frame)

    cv2.imshow("Optical flow tracking", output_frame)

    prev_gray = frame_gray.copy()  # Update previous frame for the next iteration   

    if cv2.waitKey(15) & 0xFF in [ord('q'), 27]:  # Press 'q' or 'Esc' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved successfully to: {output_video_path}")