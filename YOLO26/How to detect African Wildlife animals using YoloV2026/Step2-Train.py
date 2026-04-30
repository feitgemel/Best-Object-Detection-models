from ultralytics import YOLO

def main() :

    model = YOLO('yolo26m.pt')

    dataset_path = "Best-Object-Detection-models/YOLO26/How to detect African Wildlife animals using YoloV2026/config.yaml"

    batch_size = 16 
    project = "d:/temp/models/african-wildlife-detection"
    experiment = "My-Model-yolo26m"

    results = model.train(data=dataset_path,
                          epochs = 200 ,
                          project=project,
                          name=experiment,
                          batch=batch_size,
                          device = 0,
                          imgsz=640,
                          patience=20,
                          verbose=True,
                          val=True)
    
if __name__ == "__main__" :
        main()


