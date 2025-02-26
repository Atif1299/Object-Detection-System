import os
import subprocess
import sys

def setup_yolo():
    """
    Install and setup YOLOv5
    
    Returns:
        bool: True if setup was successful
    """
    try:
        if not os.path.exists('yolov5'):
            print("Cloning YOLOv5 repository...")
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'], check=True)
        
        # Install requirements
        print("Installing YOLOv5 requirements...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'yolov5/requirements.txt'
        ], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during YOLOv5 setup: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during YOLOv5 setup: {e}")
        return False