from ultralytics import YOLO

def main():

    #load the model 
    model = YOLO("yolov8l.yaml")

    # config file
    config_file_path = "Best-Object-Detection-models/Yolo-V8/Auto-Annotation-YoloV8-Detecting-horses/data.yaml"

    # specify output directory
    project = "C:/Data-sets/Horse-race/dataset/checkpoints"
    experiment ="My-Large-Model"

    batch_size = 16 

    # train the model

    results = model.train(data=config_file_path,
                          epochs=100,
                          project=project,
                          name=experiment,
                          batch=batch_size,
                          device=0,
                          patience=40 ,
                          imgsz = 640 ,
                          verbose=True , 
                          val=True)
    
if __name__ == "__main__":
        main()