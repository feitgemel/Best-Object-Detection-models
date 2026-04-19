from ultralytics import YOLOWorld

def main():
    model = YOLOWorld("yolov8s-world.pt")

    config_file_path = "Best-Object-Detection-models/YOLO-World/Method2-Build-Model-Based-on-Your-custom-dataset/data.yaml"

    # Specify the ouput folder for storing results 
    project = "D:/temp/models/Underwater-Trash"
    experiment = "My-YoloWorld-Model-S"

    batch_size = 16 


    # Train the model 

    results = model.train(data=config_file_path,
                          epochs=50, 
                          project=project,
                          name=experiment,
                          batch=batch_size,
                          device=0,
                          patience=5,
                          imgsz=480,
                          verbose=True,
                          val=True)
    
    print(results)

if __name__ == "__main__":
    main()