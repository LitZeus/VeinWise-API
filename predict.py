import onnxruntime as ort
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List
import os
from config import settings
import loguru
import io
from datetime import datetime

logger = loguru.logger
class VaricoseVeinsPredictor:
    def __init__(self):
        self.model = None
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_path = None
        self.load_time = None
        self.load_model()

    def load_model(self):
        try:
            # Update model path to use the correct ONNX model file
            model_path = os.path.join(os.path.dirname(__file__), "model", "varicose_detection_best.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Load ONNX model
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.model_path = model_path
            self.load_time = datetime.now().isoformat()
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image_data: bytes) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB
            image = image.convert("RGB")
            
            # Get original dimensions
            img = np.array(image)
            h0, w0 = img.shape[:2]  # original size
            
            # Resize to exact 640x640
            img_size = 640
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            
            # Convert to float32 and normalize
            img = img.astype(np.float32) / 255.0
            
            # Convert to CHW format (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            
            # Add batch dimension
            img = img[np.newaxis, :, :, :]
            
            return img, (h0, w0), (img_size, img_size)
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise

    def non_max_suppression(self, predictions: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45) -> List[np.ndarray]:
        """Apply non-max suppression to filter out overlapping boxes."""
        try:
            # Filter by confidence threshold
            mask = predictions[:, 4] > conf_thres
            predictions = predictions[mask]
            
            if len(predictions) == 0:
                return []
            
            # Sort by confidence
            indices = np.argsort(predictions[:, 4])[::-1]
            keep = []
            
            while indices.size > 0:
                i = indices[0]
                keep.append(i)
                
                # Calculate IOU with remaining boxes
                box1 = predictions[i, :4]
                box2 = predictions[indices[1:], :4]
                
                # Calculate intersection coordinates
                x1 = np.maximum(box1[0], box2[:, 0])
                y1 = np.maximum(box1[1], box2[:, 1])
                x2 = np.minimum(box1[2], box2[:, 2])
                y2 = np.minimum(box1[3], box2[:, 3])
                
                # Calculate intersection area
                w = np.maximum(0, x2 - x1)
                h = np.maximum(0, y2 - y1)
                intersection = w * h
                
                # Calculate union area
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
                union = area1 + area2 - intersection
                
                # Calculate IOU
                iou = intersection / union
                
                # Remove boxes with high IOU
                indices = indices[1:][iou < iou_thres]
            
            return predictions[keep]
        except Exception as e:
            logger.error(f"NMS error: {str(e)}")
            raise

    def postprocess(self, predictions: np.ndarray, original_shape: Tuple[int, int], new_shape: Tuple[int, int]) -> Dict[str, Any]:
        try:
            # Add debug logging
            logger.debug(f"Predictions shape: {predictions.shape}")
            
            # Reshape predictions to (5, 8400)
            predictions = predictions[0]
            
            # Apply NMS with lower threshold to get multiple detections
            predictions = self.non_max_suppression(predictions, conf_thres=0.2, iou_thres=0.3)
            
            if len(predictions) == 0:
                return {
                    "prediction": "normal",
                    "confidence": 0.0,
                    "severity": "none",
                    "boxes": [],
                    "message": "No varicose veins detected",
                    "confidence_interval": [0.0, 0.2]
                }
            
            # Get the prediction with highest confidence
            best_idx = np.argmax(predictions[:, 4])
            best_pred = predictions[best_idx]
            
            # Extract confidence
            confidence = float(best_pred[4].item())
            
            # Get bounding box coordinates
            box = best_pred[:4]
            
            # Scale boxes back to original size
            boxes = self.scale_boxes(new_shape, box, original_shape)
            
            # Determine severity based on confidence (0-100 scale)
            severity = "none"
            if confidence > 90:
                severity = "severe"
            elif confidence > 70:
                severity = "moderate"
            elif confidence > 50:
                severity = "mild"
            
            # Get confidence interval
            conf_interval = [confidence - 0.1, min(confidence + 0.1, 1.0)]
            
            # Get all detections with confidence > 0.5
            high_conf_detections = predictions[predictions[:, 4] > 0.5]
            
            return {
                "prediction": "varicose" if confidence > 0.5 else "normal",
                "confidence": confidence,
                "severity": severity,
                "boxes": boxes.tolist(),
                "message": "Varicose veins detected" if confidence > 0.5 else "No varicose veins detected",
                "confidence_interval": conf_interval,
                "num_detections": len(high_conf_detections),
                "all_detections": high_conf_detections.tolist() if len(high_conf_detections) > 0 else []
            }
        except Exception as e:
            logger.error(f"Postprocessing error: {str(e)}")
            raise

    def scale_boxes(self, input_size: Tuple[int, int], boxes: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Scale boxes from input size back to original image size."""
        h, w = input_size
        h0, w0 = original_shape
        
        # Calculate scaling ratio
        r = min(h/h0, w/w0)
        
        # Scale boxes
        boxes[[0, 2]] /= r
        boxes[[1, 3]] /= r
        
        return boxes

    def predict(self, image_data: bytes) -> Dict[str, Any]:
        try:
            if self.session is None:
                raise ValueError("Model not loaded")
                
            # Preprocess image
            processed_image, original_shape, new_shape = self.preprocess_image(image_data)
            
            # Run inference
            inputs = {self.input_name: processed_image}
            predictions = self.session.run([self.output_name], inputs)[0]
            
            # Postprocess results
            result = self.postprocess(predictions, original_shape, new_shape)
            
            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
