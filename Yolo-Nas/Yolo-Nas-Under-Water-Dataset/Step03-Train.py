import cv2
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer
from multiprocessing import freeze_support  # Import freeze_support

if __name__ == '__main__':
    freeze_support()  # Call freeze_support() in the main block

    CHECKPOINT_DIR = "C:/Data-sets/aquarium_pretrain/checkpoints"
    trainer = Trainer(experiment_name="my_custom_yolo-nas", ckpt_root_dir=CHECKPOINT_DIR)

    # Yolo-nas loaders functions
    from super_gradients.training import dataloaders
    from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val

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

    # You pass the values for dataset_params into the dataset_params argument as shown below.

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':32,
            'num_workers':2
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':32,
            'num_workers':2
        }
    )

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

    # plot a batch of training data
    # train_data.dataset.plot()

    # create in instance of the model
    from super_gradients.training import models
    model = models.get('yolo_nas_s',
                       num_classes=len(dataset_params['classes']),
                       pretrained_weights="coco"
                       )

    # Define metrics and training parameters:

    # max_epochs - Max number of training epochs
    # loss - the loss function you want to use
    # optimizer - Optimizer you will be using
    # train_metrics_list - Metrics to log during training
    # valid_metrics_list - Metrics to log during training
    # metric_to_watch - metric which the model checkpoint will be saved according to

    from super_gradients.training.losses import PPYoloELoss
    from super_gradients.training.metrics import DetectionMetrics_050
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

    train_params = {
        # ENABLING SILENT MODE
        "average_best_models":True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
        "max_epochs": 100,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            # NOTE: num_classes needs to be defined here
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                # NOTE: num_classes needs to be defined here
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }

    trainer.train(model=model,
                  training_params=train_params,
                  train_loader=train_data,
                  valid_loader=val_data)
