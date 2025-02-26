import os
import yaml
import numpy as np
import cv2
from tqdm import tqdm
from datasets import load_dataset

class DatasetPreparation:
    def __init__(self, output_dir="yolo_dataset", num_samples=1000):
        """
        Initialize dataset preparation

        Args:
            output_dir: Directory to save processed dataset
            num_samples: Number of samples to use from COCO
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.classes = None
        
    def prepare_coco_dataset(self):
        """
        Prepare COCO dataset in YOLO format
        
        Returns:
            str: Path to the data.yaml file
        """
        # Load COCO dataset
        print("Loading COCO dataset...")
        dataset = load_dataset("detection-datasets/coco", split=f"train[:{self.num_samples}]")
        
        # Create directories
        os.makedirs(f"{self.output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels/val", exist_ok=True)
        
        # Get COCO classes
        self.classes = dataset.features['objects'].feature['category'].names
        
        # Create YAML file
        yaml_content = {
            'train': f"{self.output_dir}/images/train",
            'val': f"{self.output_dir}/images/val",
            'nc': len(self.classes),
            'names': self.classes
        }
        
        with open(f"{self.output_dir}/data.yaml", 'w') as f:
            yaml.dump(yaml_content, f)
        
        # Process images and labels
        print("Processing dataset...")
        for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
            # Determine split (80% train, 20% val)
            split = 'train' if idx < int(0.8 * self.num_samples) else 'val'
            
            # Save image
            image = example['image']
            image_path = f"{self.output_dir}/images/{split}/{idx}.jpg"
            # Convert to numpy array
            image_np = np.array(image)
            # Save image using cv2
            cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            
            # Create YOLO format labels
            height, width = image.height, image.width
            labels = []
            for obj in example['objects']:
                # Convert COCO bbox to YOLO format
                bbox = obj['bbox']
                try:
                    x_center = (bbox['xmin'] + bbox['xmax']) / 2 / width
                    y_center = (bbox['ymin'] + bbox['ymax']) / 2 / height
                    w = (bbox['xmax'] - bbox['xmin']) / width
                    h = (bbox['ymax'] - bbox['ymin']) / height
                except TypeError as e:
                    print(f"Error processing bounding box: {bbox}")
                    continue  # Skip this box instead of raising an error

                class_id = obj['category']
                labels.append(f"{class_id} {x_center} {y_center} {w} {h}")
            
            # Save labels
            with open(f"{self.output_dir}/labels/{split}/{idx}.txt", 'w') as f:
                f.write('\n'.join(labels))
        
        return f"{self.output_dir}/data.yaml"