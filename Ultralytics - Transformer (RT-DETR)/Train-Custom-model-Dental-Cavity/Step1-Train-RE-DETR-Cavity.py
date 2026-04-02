from ultralytics import RTDETR
import cv2 

if __name__ == "__main__":
    # Load the model 
    model = RTDETR("rtdetr-l.pt")

    # Train the model 
    results = model.train(data="Best-Object-Detection-models/Ultralytics - Transformer (RT-DETR)/Train-Custom-model-Dental-Cavity/data.yaml",
                          epochs=100,
                          imgsz=640,
                          batch=16,
                          patience=10,
                          save=True,
                          device=0,
                          project="d:/temp/Models/RT-DETR-Cavity",
                          name="Dental-Cavity",
                          val=True)
    


    