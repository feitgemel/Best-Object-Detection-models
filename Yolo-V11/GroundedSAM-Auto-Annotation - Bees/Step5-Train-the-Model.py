from ultralytics import YOLO

def main():

    # load the model
    model = YOLO("yolo11s.pt")

    # load the data.yaml file
    config_file_path = "D:/Data-Sets-Object-Detection/Bees/dataset/data.yaml"

    # Specify the output directory for storing the result 

    project = "d:/temp/model/Bees-model"
    experiment = "Bees-small"

    batch_size = 32 # Reduce it to 16 if you have memory issues 

    # Train the model

    results = model.train( data=config_file_path,
                          epochs = 300,
                          project=project ,
                          name = experiment,
                          batch = batch_size ,
                          device = 0 ,
                          patience = 40 ,
                          imgsz=640 ,
                          verbose = True,
                          val= True)
    
if __name__ == "__main__":
    main()


