"""
Training Parameters Configuration
Centralized configuration for YOLO model training
"""

# Training hyperparameters
TRAINING_CONFIG = {
    # Basic training setup
    'epochs': 300,
    'imgsz': 640,
    'batch': 16,
    'workers': 8,
    'device': '1',  # GPU device
    'amp': True,    # Automatic Mixed Precision
    'patience': 50, # Early stopping patience
    'save': True,
    'pretrained': True,
    'verbose': True,
    'save_period': 50,  # Save checkpoint every N epochs
    
    # Optimizer settings
    'optimizer': 'AdamW',
    'lr0': 0.001,      # Initial learning rate
    'lrf': 0.01,       # Final learning rate fraction
    'momentum': 0.937,
    'weight_decay': 0.0005,
    
    # Learning rate schedule
    'cos_lr': True,    # Cosine learning rate decay
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    
    # Loss function weights
    'box': 7.5,        # Box loss weight
    'cls': 0.5,        # Classification loss weight
    'dfl': 1.5,        # Distribution focal loss weight
    
    # Regularization
    'label_smoothing': 0.1,
    
    # Data augmentation
    'hsv_h': 0.015,    # HSV-Hue augmentation
    'hsv_s': 0.7,      # HSV-Saturation augmentation
    'hsv_v': 0.4,      # HSV-Value augmentation
    'translate': 0.1,  # Translation augmentation
    'scale': 0.5,      # Scale augmentation
    'fliplr': 0.5,     # Horizontal flip probability
    'mosaic': 1.0,     # Mosaic augmentation probability
    
    # Validation settings
    'val': True,
    'plots': True,
}

# Segmentation-specific parameters
SEGMENTATION_CONFIG = {
    **TRAINING_CONFIG,  # Inherit base config
    'overlap_mask': True,    # Allow overlapping masks
    'mask_ratio': 4,         # Mask downsample ratio
    'retina_masks': False,   # Use retina masks
}

# Model-specific configurations
MODEL_CONFIGS = {
    'yolov10l': {
        **TRAINING_CONFIG,
        'model_type': 'detection',
        'architecture': 'yolov10',
        'size': 'large'
    },
    'yolov10x': {
        **TRAINING_CONFIG,
        'model_type': 'detection',
        'architecture': 'yolov10',
        'size': 'extra-large'
    },
    'yolov11l': {
        **TRAINING_CONFIG,
        'model_type': 'detection',
        'architecture': 'yolov11',
        'size': 'large'
    },
    'yolov11x': {
        **TRAINING_CONFIG,
        'model_type': 'detection',
        'architecture': 'yolov11',
        'size': 'extra-large'
    },
    'yolov8l-seg': {
        **SEGMENTATION_CONFIG,
        'model_type': 'segmentation',
        'architecture': 'yolov8',
        'size': 'large'
    },
    'yolov8x-seg': {
        **SEGMENTATION_CONFIG,
        'model_type': 'segmentation',
        'architecture': 'yolov8',
        'size': 'extra-large'
    },
    'yolov11l-seg': {
        **SEGMENTATION_CONFIG,
        'model_type': 'segmentation',
        'architecture': 'yolov11',
        'size': 'large'
    },
    'yolov11x-seg': {
        **SEGMENTATION_CONFIG,
        'model_type': 'segmentation',
        'architecture': 'yolov11',
        'size': 'extra-large'
    }
}

# Hardware requirements
HARDWARE_REQUIREMENTS = {
    'yolov10l': {'min_vram_gb': 8, 'recommended_vram_gb': 12},
    'yolov10x': {'min_vram_gb': 12, 'recommended_vram_gb': 16},
    'yolov11l': {'min_vram_gb': 8, 'recommended_vram_gb': 12},
    'yolov11x': {'min_vram_gb': 12, 'recommended_vram_gb': 16},
    'yolov8l-seg': {'min_vram_gb': 10, 'recommended_vram_gb': 14},
    'yolov8x-seg': {'min_vram_gb': 14, 'recommended_vram_gb': 18},
    'yolov11l-seg': {'min_vram_gb': 10, 'recommended_vram_gb': 14},
    'yolov11x-seg': {'min_vram_gb': 14, 'recommended_vram_gb': 18},
}

# Performance targets
PERFORMANCE_TARGETS = {
    'detection': {
        'mAP50_min': 0.85,
        'mAP50_95_min': 0.60,
        'inference_fps_min': 30
    },
    'segmentation': {
        'box_mAP50_min': 0.80,
        'mask_mAP50_min': 0.75,
        'mAP50_95_min': 0.55,
        'inference_fps_min': 25
    }
}
