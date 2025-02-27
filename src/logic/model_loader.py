from ultralytics import YOLO
import torch

def load_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return YOLO(model_path).to(device)
