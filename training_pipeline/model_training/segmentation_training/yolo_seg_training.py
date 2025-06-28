"""
YOLO Segmentation Training Pipeline
Trains YOLOv8-seg and YOLOv11-seg variants for food segmentation
"""

import os
import csv
import time
import json
from datetime import datetime
from ultralytics import YOLO
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOSegmentationTrainer:
    """YOLO Segmentation Training Manager"""
    
    def __init__(self, root_dir="/home/nagasai/", device='1'):
        self.root_dir = root_dir
        self.device = device
        self.models = ['yolov8l-seg', 'yolov8x-seg', 'yolov11l-seg', 'yolov11x-seg']
        self.data_path = os.path.join(root_dir, "custom_data.yaml")
        self.output_file = os.path.join(root_dir, 'train_seg/yolo_seg_training_metrics.csv')
        self.weights_dir = os.path.join(root_dir, 'train_seg/weights')
        self.results_dir = os.path.join(root_dir, 'train_seg/runs/train')
        
        # Create directories
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Validate data configuration
        self._validate_data_config()
    
    def _validate_data_config(self):
        """Validate dataset configuration"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data configuration not found: {self.data_path}")
        
        logger.info(f"Using dataset configuration: {self.data_path}")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Models to train: {self.models}")
    
    def safe_get(self, dict_obj, key, default=None):
        """Safely get value from dictionary with fallback keys"""
        # Try multiple key variations for segmentation metrics
        variations = [
            key,
            key.replace('(B)', ''),
            key.replace('(B)', '(M)'),  # Mask metrics
            key.replace('metrics/', ''),
            f"seg/{key}",
            f"mask/{key}"
        ]
        
        for var in variations:
            if var in dict_obj:
                return dict_obj[var]
        
        return default
    
    def train_model(self, model_name):
        """
        Train a single YOLO segmentation model
        
        Args:
            model_name: Name of the model to train (e.g., 'yolov8l-seg')
        
        Returns:
            dict: Training metrics and results
        """
        logger.info(f"Starting segmentation training for {model_name}")
        
        # Load pretrained model
        model_path = f'{model_name}.pt'  # Ultralytics will download automatically
        model = YOLO(model_path)
        
        start_time = time.time()
        
        try:
            # Training configuration for segmentation
            training_args = {
                'data': self.data_path,
                'epochs': 300,
                'imgsz': 640,
                'batch': 16,
                'workers': 8,
                'device': self.device,
                'amp': True,  # Automatic Mixed Precision
                'patience': 50,  # Early stopping patience
                'save': True,
                'project': self.results_dir,
                'name': model_name,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'label_smoothing': 0.1,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'translate': 0.1,
                'scale': 0.5,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'val': True,
                'plots': True,
                'cos_lr': True,
                'save_period': 50,  # Save checkpoint every 50 epochs
                'verbose': True,
                # Segmentation specific parameters
                'overlap_mask': True,
                'mask_ratio': 4,
                'retina_masks': False
            }
            
            # Start training
            results = model.train(**training_args)
            
            training_time = time.time() - start_time
            
            # Extract metrics
            metrics = self._extract_metrics(model_name, results, training_time)
            
            logger.info(f"Completed training for {model_name} in {training_time/3600:.2f} hours")
            logger.info(f"Best Box mAP50: {metrics.get('Box_mAP50', 'N/A')}")
            logger.info(f"Best Mask mAP50: {metrics.get('Mask_mAP50', 'N/A')}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {str(e)}")
            return self._create_error_metrics(model_name, str(e))
    
    def _extract_metrics(self, model_name, results, training_time):
        """Extract training metrics from segmentation results"""
        try:
            # Get the best results
            results_dict = results.results_dict if hasattr(results, 'results_dict') else {}
            
            # Model file path
            model_file = os.path.join(self.results_dir, model_name, 'weights', 'best.pt')
            model_size_mb = os.path.getsize(model_file) / (1024 * 1024) if os.path.exists(model_file) else None
            
            metrics = {
                'Model': model_name,
                # Box detection metrics
                'Box_mAP50': self.safe_get(results_dict, 'metrics/mAP50(B)'),
                'Box_mAP50-95': self.safe_get(results_dict, 'metrics/mAP50-95(B)'),
                'Box_Precision': self.safe_get(results_dict, 'metrics/precision(B)'),
                'Box_Recall': self.safe_get(results_dict, 'metrics/recall(B)'),
                'Box_F1': self.safe_get(results_dict, 'metrics/F1(B)'),
                # Mask segmentation metrics
                'Mask_mAP50': self.safe_get(results_dict, 'metrics/mAP50(M)'),
                'Mask_mAP50-95': self.safe_get(results_dict, 'metrics/mAP50-95(M)'),
                'Mask_Precision': self.safe_get(results_dict, 'metrics/precision(M)'),
                'Mask_Recall': self.safe_get(results_dict, 'metrics/recall(M)'),
                'Mask_F1': self.safe_get(results_dict, 'metrics/F1(M)'),
                # Loss metrics
                'Box_Loss': self.safe_get(results_dict, 'train/box_loss'),
                'Class_Loss': self.safe_get(results_dict, 'train/cls_loss'),
                'DFL_Loss': self.safe_get(results_dict, 'train/dfl_loss'),
                'Mask_Loss': self.safe_get(results_dict, 'train/seg_loss'),
                # Performance metrics
                'Inference_Speed_ms': results.speed.get('inference', None) if hasattr(results, 'speed') else None,
                'Model_Size_MB': model_size_mb,
                'Training_Time_hours': training_time / 3600,
                'Training_Status': 'Completed',
                'Timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics for {model_name}: {str(e)}")
            return self._create_error_metrics(model_name, str(e))
    
    def _create_error_metrics(self, model_name, error_msg):
        """Create metrics dict for failed training"""
        return {
            'Model': model_name,
            'Box_mAP50': None,
            'Box_mAP50-95': None,
            'Box_Precision': None,
            'Box_Recall': None,
            'Box_F1': None,
            'Mask_mAP50': None,
            'Mask_mAP50-95': None,
            'Mask_Precision': None,
            'Mask_Recall': None,
            'Mask_F1': None,
            'Box_Loss': None,
            'Class_Loss': None,
            'DFL_Loss': None,
            'Mask_Loss': None,
            'Inference_Speed_ms': None,
            'Model_Size_MB': None,
            'Training_Time_hours': None,
            'Training_Status': f'Failed: {error_msg}',
            'Timestamp': datetime.now().isoformat()
        }
    
    def train_all_models(self):
        """Train all configured segmentation models and save metrics"""
        logger.info("Starting YOLO Segmentation Training Pipeline")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        logger.info(f"CUDA Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        
        all_metrics = []
        
        # CSV file setup
        fieldnames = [
            'Model', 'Box_mAP50', 'Box_mAP50-95', 'Box_Precision', 'Box_Recall', 'Box_F1',
            'Mask_mAP50', 'Mask_mAP50-95', 'Mask_Precision', 'Mask_Recall', 'Mask_F1',
            'Box_Loss', 'Class_Loss', 'DFL_Loss', 'Mask_Loss', 'Inference_Speed_ms',
            'Model_Size_MB', 'Training_Time_hours', 'Training_Status', 'Timestamp'
        ]
        
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for model_name in self.models:
                logger.info(f"Training {model_name} ({self.models.index(model_name) + 1}/{len(self.models)})")
                
                model_metrics = self.train_model(model_name)
                all_metrics.append(model_metrics)
                
                # Write metrics immediately
                writer.writerow(model_metrics)
                csvfile.flush()  # Ensure data is written
                
                logger.info(f"Finished training {model_name}")
                logger.info("-" * 50)
        
        # Save summary JSON
        summary_file = os.path.join(os.path.dirname(self.output_file), 'yolo_seg_training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'training_config': {
                    'models': self.models,
                    'data_path': self.data_path,
                    'device': self.device,
                    'total_models': len(self.models)
                },
                'results': all_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Segmentation training pipeline completed!")
        logger.info(f"Metrics saved to: {self.output_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print summary
        self._print_training_summary(all_metrics)
        
        return all_metrics
    
    def _print_training_summary(self, metrics_list):
        """Print training summary"""
        logger.info("\n" + "="*70)
        logger.info("SEGMENTATION TRAINING SUMMARY")
        logger.info("="*70)
        
        for metrics in metrics_list:
            model_name = metrics['Model']
            status = metrics['Training_Status']
            
            if status == 'Completed':
                logger.info(f"{model_name}:")
                logger.info(f"  Box mAP50: {metrics['Box_mAP50']:.4f}" if metrics['Box_mAP50'] else "  Box mAP50: N/A")
                logger.info(f"  Mask mAP50: {metrics['Mask_mAP50']:.4f}" if metrics['Mask_mAP50'] else "  Mask mAP50: N/A")
                logger.info(f"  Training Time: {metrics['Training_Time_hours']:.2f}h" if metrics['Training_Time_hours'] else "  Training Time: N/A")
                logger.info(f"  Model Size: {metrics['Model_Size_MB']:.1f}MB" if metrics['Model_Size_MB'] else "  Model Size: N/A")
            else:
                logger.info(f"{model_name}: {status}")
            
            logger.info("-" * 40)

def main():
    """Main execution function"""
    # Configuration
    trainer = YOLOSegmentationTrainer(
        root_dir="/path/to/root/directory",  # Change to your root directory
        device='1'  # GPU device
    )
    
    # Run training pipeline
    results = trainer.train_all_models()
    
    return results

if __name__ == "__main__":
    main()
