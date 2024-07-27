from ultralytics import YOLO 

def main():

    #load the model 
    model = YOLO("yolov8n.yaml")

    # use the config file 
    data_yaml_file = "C:/Data-sets/Playing Cards/data.yaml"

    project = "C:/Data-sets/Playing Cards"
    experiment = "My-Card-Model"

    batch_size = 32

    # train the model 
    results = model.train(data=data_yaml_file,
                          epochs=50,
                          project=project, 
                          name = experiment , 
                          batch = batch_size , 
                          device = 0 ,
                          patience = 5, 
                          imgsz=640 , 
                          verbose = True ,
                          val=True)

    # The results will be store in :  C:/Data-sets/Playing Cards/y-Card-Model


if __name__ == "__main__":
    main()
