import cv2
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training import Trainer
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

import multiprocessing

def main():
    # load your dataset parameters into a dictionary
    dataset_params = {
            'data_dir':'C:/Data-sets/aquarium_pretrain',
            'train_images_dir':'train/images',
            'train_labels_dir':'train/labels',
            'val_images_dir':'valid/images',
            'val_labels_dir':'valid/labels',
            'test_images_dir':'test/images',
            'test_labels_dir':'test/labels',
            'classes': ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
        }

    test_data = coco_detection_yolo_format_val(
            dataset_params={
                'data_dir': dataset_params['data_dir'],
                'images_dir': dataset_params['test_images_dir'],
                'labels_dir': dataset_params['test_labels_dir'],
                'classes': dataset_params['classes']
            },
            dataloader_params={
                'batch_size':32,
                'num_workers':2
            }
        )

    CHECKPOINT_DIR = "C:/Data-sets/aquarium_pretrain/checkpoints"
    trainer = Trainer(experiment_name="my_custom_yolo-nas", ckpt_root_dir=CHECKPOINT_DIR)

    bestWeights = "C:/Data-sets/aquarium_pretrain/checkpoints/my_custom_yolo-nas/RUN_20240809_115703_065907/ckpt_best.pth"

    best_model = models.get('yolo_nas_s',
                            num_classes=len(dataset_params['classes']),
                            checkpoint_path=bestWeights)

    # run a predicition on the full test folder    
    result = trainer.test(model=best_model,
                test_loader=test_data,
                test_metrics_list=DetectionMetrics_050(score_thres=0.2, 
                                                       top_k_predictions=300, 
                                                       num_cls=len(dataset_params['classes']), 
                                                       normalize_targets=True, 
                                                       post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                              nms_top_k=1000, 
                                                                                                              max_predictions=300,                                                                              
                                                                                                              nms_threshold=0.7)
                                                      ))
    print(result)
    
    # run a predicition on speciifc image 
    jellyfish = "Best-Object-Detection-models/Yolo-Nas/Yolo-Nas-Under-Water-Dataset/JellyFish.jpg"
    starfish =  "Best-Object-Detection-models/Yolo-Nas/Yolo-Nas-Under-Water-Dataset/StarFish2.jpg"
    
    img = cv2.imread(jellyfish)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)    


    predict_results = best_model.predict(img)
    print(predict_results)
    predict_results.show()
    
 



    

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
