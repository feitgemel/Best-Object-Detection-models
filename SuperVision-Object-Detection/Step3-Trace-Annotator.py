import supervision as sv 
from ultralytics import YOLO

model = YOLO("yolov8x.pt") 

trace_annotator = sv.TraceAnnotator() 

video_info = sv.VideoInfo.from_video_path(video_path="Best-Object-Detection-models/SuperVision-Object-Detection/test.mp4")
frames_generator = sv.get_video_frames_generator(source_path="Best-Object-Detection-models/SuperVision-Object-Detection/test.mp4")

tracker = sv.ByteTrack() 

with sv.VideoSink(target_path="Best-Object-Detection-models/SuperVision-Object-Detection/Trace_annotator_output.mp4",video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        annotated_frame = trace_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        

        sink.write_frame(annotated_frame)


