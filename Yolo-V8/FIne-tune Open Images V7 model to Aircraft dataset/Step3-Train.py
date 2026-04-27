from ultralytics import YOLO

def main():

    # Load the pre-trained weights ofr the OpenImages V7 model 
    path_model = "D:/Temp/Models/yolov8l-oiv7.pt" # This points to the downloaded weights 

    model = YOLO(path_model)

    config_file_path = "Best-Object-Detection-models/Yolo-V8/FIne-tune Open Images V7 model to Aircraft dataset/data.yaml"

    # Output folder

    project = "D:/Temp/Models/Military Aircraft Detection"
    experiment = "My-Model-224"
    batch_size = 16 # Reduce it if you have memory issues  

    # Train the model 
    results = model.train(data=config_file_path,
                          epochs=200,
                          project=project,
                          name=experiment,
                          batch=batch_size,
                          device=0, 
                          patience=30,
                          imgsz=224,
                          verbose=True,
                          val=True)
    


if __name__ == "__main__":
    main()


