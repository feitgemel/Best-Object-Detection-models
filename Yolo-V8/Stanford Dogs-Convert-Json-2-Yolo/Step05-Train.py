from ultralytics import YOLO

def main():

    model = YOLO("yolov8n.yaml")

    # config file
    config_file_path = "Best-Object-Detection-models/Yolo-V8/Stanford Dogs-Convert-Json-2-Yolo/data.yaml"

    # Storing results
    project = "C:/Data-sets/Stanford Dogs Dataset/dataset"
    experiment_name = "Nano-Model"

    batch_size = 16 

    # train the model

    results = model.train(data=config_file_path,
                          epochs=100,
                          project=project,
                          name=experiment_name,
                          batch=batch_size,
                          device=0,
                          patience=10 ,
                          imgsz=640,
                          verbose=True,
                          val=True)
    
if __name__ == "__main__":
        main()