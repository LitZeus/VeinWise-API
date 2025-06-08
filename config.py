from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "VeinWise API"
    
    # Model Settings
    MODEL_PATH: str = "model/varicose_detection_best.onnx"
    
    # CORS Settings
    allowed_origins: List[str] = ["*"]
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
