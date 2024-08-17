from ultralytics import YOLO

def main():

    #load the Yolo model
    model = YOLO('yolov8s.yaml') # load the small model

    # use the yaml file 
    config_file_path = "Best-Object-Detection-models/Yolo-V8/Auto-Annotation-FromYotube-Maccabi/data.yaml"

    project = "C:/Data-sets/Mac-Real/dataset/checkpoints"
    experiment = "small-Model"

    batch_size = 32 # reduce to 16 if you have memory errors

    # train the model

    results  = model.train(data=config_file_path,
                           epochs=300,
                           project=project,
                           name=experiment,
                           batch=batch_size,
                           device=0,
                           patience=40,
                           imgsz=640,
                           verbose=True,
                           val=True)
    

if __name__ =="__main__":
        main()