import os
import subprocess
import sys

def train_custom_model(data_yaml_path, epochs=100, batch_size=16, img_size=640, weights='yolov5s.pt'):
    """
    Train YOLOv5 on custom dataset
    
    Args:
        data_yaml_path: Path to data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
        weights: Initial weights to use
    
    Returns:
        str: Path to best trained weights
    """
    try:
        print(f"Starting YOLOv5 training with {epochs} epochs, batch size {batch_size}...")
        
        train_command = [
            sys.executable,
            "yolov5/train.py",
            "--img", str(img_size),
            "--batch", str(batch_size),
            "--epochs", str(epochs),
            "--data", data_yaml_path,
            "--weights", weights
        ]
        
        subprocess.run(train_command, check=True)
        
        exp_dirs = [d for d in os.listdir("yolov5/runs/train/") if d.startswith("exp")]
        
        if not exp_dirs:
            raise FileNotFoundError("No training output directories found")
            
        latest_exp = max(exp_dirs, key=lambda x: int(x.replace("exp", "")) if x != "exp" else 1)
        weights_path = f"yolov5/runs/train/{latest_exp}/weights/best.pt"
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
            
        print(f"Training complete. Best weights saved to: {weights_path}")
        return weights_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        raise