from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from loguru import logger
from typing import Dict, Any
import uvicorn
from enum import Enum
from datetime import datetime
from PIL import Image
import io
import numpy as np
import psutil

from config import settings
from predict import VaricoseVeinsPredictor
from visualize import draw_bounding_boxes

app = FastAPI(title="Varicose Veins Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = VaricoseVeinsPredictor()

class PredictionType(str, Enum):
    json = "json"
    image = "image"

@app.get("/")
async def root():
    return {
        "message": "Welcome to Varicose Veins Detection API",
        "endpoints": {
            "/api/v1/predict": "Make prediction on uploaded image",
            "/api/v1/visualize": "Get image with bounding boxes",
            "/api/v1/health": "Check API health",
            "/api/v1/metrics": "Get model metrics"
        }
    }

@app.post("/api/v1/predict")
async def predict_varicose_veins(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be PNG, JPG, or JPEG")
        
        # Read image file
        contents = await file.read()
        
        # Make prediction
        result = predictor.predict(contents)
        
        # Calculate severity based on confidence (0-100 scale)
        severity = "none"
        if result["confidence"] > 90:
            severity = "severe"
        elif result["confidence"] > 70:
            severity = "moderate"
        elif result["confidence"] > 50:
            severity = "mild"
        
        # Format the response
        response = {
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 2),  # Round to 2 decimal places
            "severity": severity,
            "detection_status": "Varicose veins detected" if result["prediction"] == "varicose" else "No varicose veins detected",
            "bounding_boxes": result["boxes"],
            "num_detections": result["num_detections"],
            "model_version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/visualize")
async def visualize_prediction(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be PNG, JPG, or JPEG")
        
        # Read image file
        contents = await file.read()
        
        # Make prediction
        result = predictor.predict(contents)
        
        # Convert bytes to PIL image
        image = Image.open(io.BytesIO(contents))
        
        # Draw bounding boxes
        return draw_bounding_boxes(
            image=image,
            boxes=result["boxes"],
            prediction=result["prediction"],
            confidence=result["confidence"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    try:
        if predictor.session is None:
            return {
                "status": "unhealthy",
                "message": "Model not loaded",
                "model_loaded": False
            }
        
        # Test model inference
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        test_input = test_image.astype(np.float32) / 255.0
        test_input = np.transpose(test_input, (2, 0, 1))
        test_input = test_input[np.newaxis, :, :, :]
        
        inputs = {predictor.input_name: test_input}
        _ = predictor.session.run([predictor.output_name], inputs)
        
        return {
            "status": "healthy",
            "message": "Model loaded and tested successfully",
            "model_loaded": True
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"Error during health check: {str(e)}",
            "model_loaded": False
        }

@app.get("/api/v1/metrics")
async def get_metrics():
    try:
        # Get model input/output shapes
        input_shape = predictor.session.get_inputs()[0].shape
        output_shape = predictor.session.get_outputs()[0].shape
        
        return {
            "model_metrics": {
                "input_shape": input_shape,
                "output_shape": output_shape,
                "model_path": predictor.model_path,
                "last_load_time": predictor.load_time
            },
            "system_metrics": {
                "total_memory": psutil.virtual_memory().total,
                "available_memory": psutil.virtual_memory().available,
                "cpu_count": psutil.cpu_count()
            }
        }
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
