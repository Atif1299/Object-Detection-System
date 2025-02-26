import torch
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='yolov5/yolov5s.pt', conf_thresh=0.25):
        """
        Initialize object detector
        
        Args:
            model_path: Path to trained model weights
            conf_thresh: Confidence threshold for detections
        """
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            self.model.conf = conf_thresh
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_image(self, image_path):
        """
        Detect objects in single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            tuple: (detection results, annotated image)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = self.model(img)
            
            detections = results.pandas().xyxy[0]
            
            annotated_img = img.copy()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for cv2
            
            for idx, row in detections.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = f"{row['name']} {row['confidence']:.2f}"
                
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            return results, annotated_img
            
        except Exception as e:
            print(f"Error during image detection: {e}")
            raise
    
    def detect_video(self, video_path, output_path):
        """
        Detect objects in video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            
        Returns:
            str: Path to output video
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            print("Processing video frames...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % 20 == 0:  
                    print(f"Processed {frame_count} frames")
                
                results = self.model(frame)
                annotated_frame = np.squeeze(results.render())
                
                out.write(annotated_frame)
                
            cap.release()
            out.release()
            print(f"Video processing complete. Saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error during video detection: {e}")
            raise