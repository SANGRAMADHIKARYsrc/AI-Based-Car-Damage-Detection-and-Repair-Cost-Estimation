"""Data Augmentation Configurations and Albumentations/YOLO Policies
Specifically tailored for automotive damage inspection:
1. Minority class oversampling and targeted copy-paste augmentation.
2. Photometric jitter (glare, shadow, overcast, dusk reflection).
3. Geometric transforms (scale variation, perspective, horizontal flips).
"""

from typing import Dict, Any


def get_yolo_augmentation_config() -> Dict[str, Any]:
    """Returns standard Ultralytics YOLO augmentation hyperparameters optimized for CarDD."""
    return {
        # Photometric jitter
        "hsv_h": 0.015,       # image HSV-Hue augmentation (fraction)
        "hsv_s": 0.7,         # image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.4,         # image HSV-Value augmentation (fraction)
        
        # Geometric augmentations
        "degrees": 10.0,      # image rotation (+/- deg)
        "translate": 0.1,     # image translation (+/- fraction)
        "scale": 0.5,         # image scale (+/- gain)
        "shear": 2.0,         # image shear (+/- deg)
        "perspective": 0.0005,# image perspective (+/- fraction)
        
        # Flips (only horizontal - cars don't drive upside down)
        "fliplr": 0.5,        # horizontal flip probability
        "flipud": 0.0,        # vertical flip disabled
        
        # Advanced multi-image compositing (critical for small/rare classes like tire flat & glass)
        "mosaic": 1.0,        # mosaic probability (combines 4 images)
        "mixup": 0.15,        # mixup probability (blends 2 images)
        "copy_paste": 0.3,    # copy-paste augmentation probability
        
        # Erasing / dropout
        "erasing": 0.2        # random erasing probability for occlusion robustness
    }


def get_severity_augmentation_config() -> Dict[str, Any]:
    """Augmentation configuration for 3-class severity CNN/ViT training."""
    return {
        "resize": (224, 224),
        "random_crop": (224, 224),
        "horizontal_flip_prob": 0.5,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1
        },
        "affine": {
            "degrees": (-15, 15),
            "translate": (0.05, 0.05),
            "scale": (0.9, 1.1)
        },
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
