from ultralytics import YOLO

def main():

    #load the model
    model = YOLO("yolov8l.yaml")

    # use the config file
    config_file_path = "Best-Object-Detection-models/Yolo-V8/Bone Fracture Detection/data.yaml"

    project = "C:/Data-sets/Bone fracture detection"
    experiment = "My-Model"

    batch_size = 32 # redcue it to 16 if you have memory issuses

    # train

    result = model.train(data=config_file_path,
                         epochs=1000,
                         project=project,
                         name=experiment,
                         batch=batch_size,
                         device=0,
                         patience=300,
                         imgsz=350,
                         verbose=True,
                         val=True)
    

if __name__ == "__main__":
    main()