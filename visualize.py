import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict, Any, Tuple
from fastapi.responses import StreamingResponse

def draw_bounding_boxes(image: Image.Image, boxes: list, prediction: str, confidence: float) -> StreamingResponse:
    """
    Draw bounding boxes on the image and return it as a streaming response.
    """
    try:
        # Convert PIL image to OpenCV format
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw rectangle
            color = (0, 255, 0) if prediction == "normal" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{prediction} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert back to RGB for PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image
        result = Image.fromarray(img)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/jpeg")
        
    except Exception as e:
        raise Exception(f"Error visualizing image: {str(e)}")
