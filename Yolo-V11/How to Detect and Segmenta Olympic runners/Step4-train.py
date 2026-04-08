from ultralytics import YOLO

def main():

    # load the model
    model = YOLO("yolo11n.pt") 

    # Use the model 
    config_file_path = "D:/Data-Sets-Object-Detection/Athletic-Runner/Dataset/data.yaml"

    # Specify the output directory 
    project = "d:/temp/models/Athletic-Runner/checkpoits"
    experiment = "Athletic-Runner-nano"

    # Train the model

    results = model.train( data=config_file_path,
                          epochs = 100 ,
                          project=project ,
                          name = experiment,
                          batch = 16 ,
                          device = 0 ,
                          patience = 14, 
                          imgsz = 640,
                          verbose = True,
                          val= True)
    
if __name__ == "__main__":
    main()


    