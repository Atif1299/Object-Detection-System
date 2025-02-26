import os
import argparse
import matplotlib.pyplot as plt
import cv2

from dataset_preparation import DatasetPreparation
from yolo_setup import setup_yolo
from train import train_custom_model
from object_detector import ObjectDetector

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Object Detection with YOLOv5")
    parser.add_argument("--samples", type=int, default=100, help="Number of COCO samples to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for training")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and use existing weights")
    parser.add_argument("--video", type=str, help="Path to input video file for detection")
    
    return parser.parse_args()

def main():
    """Main function to run the complete pipeline"""
    args = parse_args()
    
    print("Setting up YOLO...")
    if not setup_yolo():
        print("Failed to set up YOLO. Exiting.")
        return
    
    dataset_dir = "yolo_dataset"
    output_dir = "detection_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    weights_path = "yolov5/runs/train/exp/weights/best.pt"
    
    if not args.skip_training:
        print(f"Preparing dataset with {args.samples} samples...")
        dataset_prep = DatasetPreparation(output_dir=dataset_dir, num_samples=args.samples)
        data_yaml_path = dataset_prep.prepare_coco_dataset()
        
        print("Training model...")
        weights_path = train_custom_model(
            data_yaml_path, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            img_size=args.img_size
        )
    else:
        print("Skipping training as requested.")
        if not os.path.exists(weights_path):
            print(f"Warning: Weights file not found at {weights_path}. Using default YOLOv5s weights.")
            weights_path = "yolov5s.pt"
    
    print(f"Initializing detector with weights: {weights_path}")
    detector = ObjectDetector(model_path=weights_path, conf_thresh=args.conf)
    
    if os.path.exists(f"{dataset_dir}/images/val") and len(os.listdir(f"{dataset_dir}/images/val")) > 0:
        test_image_path = f"{dataset_dir}/images/val/{os.listdir(f'{dataset_dir}/images/val')[0]}"
        print(f"Testing on validation image: {test_image_path}")
        
        results, annotated_img = detector.detect_image(test_image_path)
        
        result_path = f"{output_dir}/detection_result.jpg"
        cv2.imwrite(result_path, annotated_img)
        
        print(f"Detection result saved to: {result_path}")
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Object Detection Result")
        plt.show()
    
    if args.video and os.path.exists(args.video):
        print(f"Processing video: {args.video}")
        output_video = f"{output_dir}/processed_video.mp4"
        detector.detect_video(args.video, output_video)
        print(f"Processed video saved to: {output_video}")
    
    print("Object detection pipeline completed successfully!")

if __name__ == "__main__":
    main()