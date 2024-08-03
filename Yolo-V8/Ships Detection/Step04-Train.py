from ultralytics import YOLO 

def main() :

    #load the model 
    model = YOLO("yolov8l.yaml")

    # config file 
    config_file_path = "Best-Object-Detection-models/Yolo-V8/Ships Detection/data.yaml"

    # Specify the output folder for the results
    project = "C:/Data-sets/Ships detection"
    experiment = "My-model"

    batch_size = 16 
    # train 

    results = model.train(data=config_file_path,
                          epochs=100,
                          name=experiment,
                          project=project,
                          batch=batch_size,
                          device = 0,
                          patience = 10 ,
                          imgsz=640 ,
                          verbose=True,
                          val=True)
    


# The results will be stored in "C:/Data-sets/Ships detection/My-model"

if __name__ == "__main__":
    main()