# Object Detection System

A modular system for object detection in images and videos using YOLOv5. This project allows you to train custom object detection models using the COCO dataset and use them for inference on images and videos.

## Project Structure

```
object-detection-system/
│
├── main.py                # Main script to run the complete pipeline
├── dataset_preparation.py # Handles dataset preparation
├── yolo_setup.py          # Sets up YOLOv5 environment
├── object_detector.py     # Contains the ObjectDetector class
├── train.py               # Handles model training
├── requirements.txt       # Dependencies
│
├── yolo_dataset/          # Generated dataset directory
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml          # Dataset configuration
│
├── yolov5/                # YOLOv5 repository (cloned during setup)
│
└── detection_outputs/     # Directory for detection results
    ├── detection_result.jpg
    └── processed_video.mp4
```

## Features

- Automatic YOLOv5 setup and installation
- COCO dataset preparation in YOLO format
- Custom model training with configurable parameters
- Object detection for images with bounding box visualization
- Video processing with object detection
- Modular architecture for easy extension and maintenance

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/object-detection-system.git
   cd object-detection-system
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. The system will automatically clone and set up YOLOv5 on first run.

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```
python main.py
```

This will:
1. Set up YOLOv5
2. Download and prepare COCO dataset samples
3. Train a custom model
4. Test the model on a validation image
5. Display and save the detection results

### Command-line Options

Customize execution with these options:

```
python main.py --samples 200 --epochs 10 --batch-size 16
```

Available options:
- `--samples`: Number of COCO samples to use (default: 100)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size for training (default: 8)
- `--img-size`: Image size for training (default: 640)
- `--conf`: Confidence threshold for detection (default: 0.25)
- `--skip-training`: Skip training and use existing weights
- `--video`: Path to input video file for detection

### Process a Video

To process a video with an existing model:

```
python main.py --skip-training --video path/to/your/video.mp4
```

## Components

### Dataset Preparation (`dataset_preparation.py`)

The `DatasetPreparation` class handles:
- Downloading COCO dataset samples
- Converting annotations to YOLO format
- Creating train/validation splits
- Generating a YAML configuration file

### YOLOv5 Setup (`yolo_setup.py`)

Sets up the YOLOv5 environment by:
- Cloning the YOLOv5 repository
- Installing required dependencies

### Model Training (`train.py`)

Handles the training process with:
- Configurable epochs, batch size, and image size
- Training progress tracking
- Best weights selection

### Object Detector (`object_detector.py`)

The `ObjectDetector` class provides:
- Image detection with bounding box visualization
- Video processing with frame-by-frame detection
- Confidence thresholding

## Requirements

- Python 3.8+
- PyTorch 1.7+
- OpenCV
- Other dependencies listed in requirements.txt

## Example Results

After running the system, you can find:
- Annotated images in `detection_outputs/detection_result.jpg`
- Processed videos in `detection_outputs/processed_video.mp4`

## Advanced Usage

### Using a Pre-trained Model

To use a pre-trained model without retraining:

```python
from object_detector import ObjectDetector

detector = ObjectDetector(model_path='path/to/weights.pt', conf_thresh=0.3)

results, annotated_img = detector.detect_image('path/to/image.jpg')

detector.detect_video('input_video.mp4', 'output_video.mp4')
```

### Custom Dataset

The system is designed to work with COCO dataset by default, but you can modify the `DatasetPreparation` class to work with your own dataset.

## Troubleshooting

- **CUDA out of memory**: Reduce batch size using `--batch-size`
- **Slow training**: Try reducing the number of samples with `--samples`
- **Poor detection results**: Train for more epochs with `--epochs` or adjust confidence threshold with `--conf`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- COCO dataset
