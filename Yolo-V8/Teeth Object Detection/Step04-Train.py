from ultralytics import YOLO

def main() :

    # load the model 
    model = YOLO("yolov8l.yaml")

    # load the yaml file 
    config_file_path = "C:/Data-sets/teeth2/data.yaml"

    # specify output folder to store the results

    project = "C:/Data-sets/teeth2"
    experiment = "My-Teeth-Model"

    batch_size = 16 

    # train the model 

    results = model.train(data = config_file_path, 
                          epochs = 100 ,
                          project = project, 
                          name = experiment, 
                          batch = batch_size, 
                          device = 0,
                          val=True )
    
    # the results will be saved to "C:/Data-sets/teeth2/My-Teeth-Model"

if __name__ == "__main__" :
    main()

                          
                          
                          
                          
                          
                          