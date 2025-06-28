"""
YOLO Bounding Box Detection Training Pipeline
Trains YOLOv10 and YOLOv11 variants for food detection
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

class YOLOBoundingBoxTrainer:
    """YOLO Bounding Box Detection Training Manager"""
    
    def __init__(self, root_dir="/home/nagasai/", device='1'):
        self.root_dir = root_dir
        self.device = device
        self.models = ['yolov10l', 'yolov10x', 'yolov11l', 'yolov11x']
        self.data_path = os.path.join(root_dir, "custom_data.yaml")
        self.output_file = 'yolo_bb_training_metrics.csv'
        self.weights_dir = os.path.join(root_dir, 'train/weights')
        self.results_dir = os.path.join(root_dir, 'runs/train')
        
        # Create directories
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
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
        return dict_obj.get(key, dict_obj.get(key.replace('(B)', ''), default))
    
    def train_model(self, model_name):
        """
        Train a single YOLO model
        
        Args:
            model_name: Name of the model to train (e.g., 'yolov10l')
        
        Returns:
            dict: Training metrics and results
        """
        logger.info(f"Starting training for {model_name}")
        
        # Load pretrained model
        model_path = f'{model_name}.pt'  # Ultralytics will download automatically
        model = YOLO(model_path)
        
        start_time = time.time()
        
        try:
            # Training configuration
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
                'verbose': True
            }
            
            # Start training
            results = model.train(**training_args)
            
            training_time = time.time() - start_time
            
            # Extract metrics
            metrics = self._extract_metrics(model_name, results, training_time)
            
            logger.info(f"Completed training for {model_name} in {training_time/3600:.2f} hours")
            logger.info(f"Best mAP50: {metrics.get('mAP50', 'N/A')}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {str(e)}")
            return self._create_error_metrics(model_name, str(e))
    
    def _extract_metrics(self, model_name, results, training_time):
        """Extract training metrics from results"""
        try:
            # Get the best results
            results_dict = results.results_dict if hasattr(results, 'results_dict') else {}
            
            # Model file path
            model_file = os.path.join(self.results_dir, model_name, 'weights', 'best.pt')
            model_size_mb = os.path.getsize(model_file) / (1024 * 1024) if os.path.exists(model_file) else None
            
            metrics = {
                'Model': model_name,
                'mAP50': self.safe_get(results_dict, 'metrics/mAP50(B)'),
                'mAP50-95': self.safe_get(results_dict, 'metrics/mAP50-95(B)'),
                'Precision': self.safe_get(results_dict, 'metrics/precision(B)'),
                'Recall': self.safe_get(results_dict, 'metrics/recall(B)'),
                'F1-Score': self.safe_get(results_dict, 'metrics/F1(B)'),
                'Box Loss': self.safe_get(results_dict, 'train/box_loss'),
                'Class Loss': self.safe_get(results_dict, 'train/cls_loss'),
                'DFL Loss': self.safe_get(results_dict, 'train/dfl_loss'),
                'Inference Speed (ms)': results.speed.get('inference', None) if hasattr(results, 'speed') else None,
                'Model Size (MB)': model_size_mb,
                'Training Time (hours)': training_time / 3600,
                'Training Status': 'Completed',
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
            'mAP50': None,
            'mAP50-95': None,
            'Precision': None,
            'Recall': None,
            'F1-Score': None,
            'Box Loss': None,
            'Class Loss': None,
            'DFL Loss': None,
            'Inference Speed (ms)': None,
            'Model Size (MB)': None,
            'Training Time (hours)': None,
            'Training Status': f'Failed: {error_msg}',
            'Timestamp': datetime.now().isoformat()
        }
    
    def train_all_models(self):
        """Train all configured models and save metrics"""
        logger.info("Starting YOLO Bounding Box Training Pipeline")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        logger.info(f"CUDA Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        
        all_metrics = []
        
        # CSV file setup
        fieldnames = [
            'Model', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score',
            'Box Loss', 'Class Loss', 'DFL Loss', 'Inference Speed (ms)', 
            'Model Size (MB)', 'Training Time (hours)', 'Training Status', 'Timestamp'
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
        summary_file = 'yolo_bb_training_summary.json'
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
        
        logger.info(f"Training pipeline completed!")
        logger.info(f"Metrics saved to: {self.output_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print summary
        self._print_training_summary(all_metrics)
        
        return all_metrics
    
    def _print_training_summary(self, metrics_list):
        """Print training summary"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        
        for metrics in metrics_list:
            model_name = metrics['Model']
            status = metrics['Training Status']
            
            if status == 'Completed':
                logger.info(f"{model_name}:")
                logger.info(f"  mAP50: {metrics['mAP50']:.4f}" if metrics['mAP50'] else "  mAP50: N/A")
                logger.info(f"  mAP50-95: {metrics['mAP50-95']:.4f}" if metrics['mAP50-95'] else "  mAP50-95: N/A")
                logger.info(f"  Training Time: {metrics['Training Time (hours)']:.2f}h" if metrics['Training Time (hours)'] else "  Training Time: N/A")
                logger.info(f"  Model Size: {metrics['Model Size (MB)']:.1f}MB" if metrics['Model Size (MB)'] else "  Model Size: N/A")
            else:
                logger.info(f"{model_name}: {status}")
            
            logger.info("-" * 30)

def main():
    """Main execution function"""
    # Configuration
    trainer = YOLOBoundingBoxTrainer(
        root_dir="/path/to/root/directory",  # Change to your root directory
        device='1'  # GPU device
    )
    
    # Run training pipeline
    results = trainer.train_all_models()
    
    return results

if __name__ == "__main__":
    main()
