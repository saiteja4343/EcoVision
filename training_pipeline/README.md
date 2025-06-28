# 🤖 YOLO Training Pipeline Documentation

This directory contains the complete documentation and code for training YOLO models for food recognition in the EcoVision AI project.

## 📋 Pipeline Overview

The training pipeline consists of multiple steps to create high-quality datasets and train both detection and segmentation models:

### **Step 1: Data Processing & Dataset Creation**
- **Roboflow Dataset**: Initial bounding box annotations from curated image collection
- **SAM 2.1 Conversion**: Automatic conversion from bounding boxes to segmentation masks
- **Synthetic Data Generation**: Flux 1.dev for generating additional training images
- **Lang-SAM Segmentation**: Automatic segmentation for synthetic images
- **Data Preprocessing**: Balancing, filtering, and train/test/valid splitting

### **Step 2: Model Training** (Coming Next)
- YOLO detection model training
- YOLO segmentation model training
- Model evaluation and validation

## 📁 Directory Structure

```
📁 training_pipeline/
├── README.md                           # Main training pipeline overview
├── 📁 data_processing/
│   ├── README.md                       # Step 1 documentation
│   ├── 📁 roboflow_to_segmentation/
│   │   └── sam_bbox_to_segmentation.py # SAM 2.1 conversion script
│   ├── 📁 synthetic_data_generation/
│   │   ├── flux_image_generation.py    # Flux 1.dev generation script
│   │   ├── lang_sam_segmentation.py    # Lang-SAM segmentation script
│   │   └── README.md                   # Synthetic data explanation
│   └── 📁 data_preprocessing/
│       ├── dataset_balancing.py        # 500 images per class script
│       ├── train_test_split.py         # Data splitting script
│       └── dataset_statistics.py       # File counting script
├── 📁 model_training/
├── README.md                                    # Main Step 2 documentation
├── 📁 bounding_box_training/
│   ├── yolo_bb_training.py                    # YOLOv10/v11 detection training script
├── 📁 segmentation_training/
│   ├── yolo_seg_training.py                   # YOLOv8/v11-seg training script
└── 📁 configs/
    ├── custom_data.yaml                       # Dataset configuration
    └── training_params.py                     # Training hyperparameters

```

## 🎯 Dataset Information

**Food Classes**: 45 categories

```
Apple, Asparagus, Avocado, Banana, Beans, Beetroot, Bell_pepper, Blueberry,
Broccoli, Brussel_sprouts, Cabbage, Carrot, Cauliflower, Celery, Cucumber,
Eggplant, Galia, Garlic, Ginger, Grapefruit, Grapes, Kaki, Kiwi, Lemon,
Lettuce, Mango, Melon, Mushroom, Onion, Orange, Passion_fruit, Peach, Pear,
Peas, Pineapple, Plum, Pomegranate, Potato, Pumpkin, Radish, Raspberry,
Strawberry, Tomato, Watermelon, Zucchini
```


**Target Dataset Size**: ~500 images per class (22,500 total images)

**Data Sources**:
- **Real Images**: Roboflow dataset with manual annotations
- **Synthetic Images**: Flux 1.dev generated images with custom prompts
- **Segmentation**: SAM 2.1 + Lang-SAM for automated mask generation

## 🔗 External Resources

- **Roboflow Dataset**: [Food Recognition Dataset](https://universe.roboflow.com/yolov8dataset-8uxqu/f_v_added_2)
- **SAM 2.1**: [Segment Anything Model 2.1](https://github.com/facebookresearch/segment-anything-2)
- **Flux 1.dev**: [Text-to-Image Generation](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- **Lang-SAM**: [Language Segment Anything](https://github.com/luca-medeiros/lang-segment-anything)

## 🚀 Quick Start

1. **Follow Step 1**: Data processing and dataset creation
2. **Follow Step 2**: Model training (coming next)
3. **Use trained models**: In the main EcoVision AI application

Each step contains detailed documentation and ready-to-use scripts.

---
**Created by**: Naga Sai Teja Kolakaleti  
**Organization**: Kuenneth Research Group, University of Bayreuth