# VeinWise-API

A FastAPI-based backend for varicose veins detection and patient management. This project provides RESTful endpoints for image prediction, patient data handling, and health checks. Designed for easy integration with modern web frontends (e.g., Next.js on Vercel).

## Features
- Predict varicose veins from medical images using ONNX models
- Patient registration and management endpoints
- Health and metrics endpoints for monitoring
- CORS support for modern frontend frameworks

## Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/VeinWise-API.git
   cd VeinWise-API
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Configure environment variables:**
   - Copy `.env.example` to `.env` and set your values (CORS, model path, etc).

## Usage
- **Development:**
  ```sh
  uvicorn main:app --reload
  ```
- **Production:**
  ```sh
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
  ```
- **Docker:**
  ```sh
  docker build -t veinwise-api .
  docker run -p 8000:8000 --env-file .env veinwise-api
  ```

## API Endpoints
- `POST /api/v1/predict` — Predict varicose veins from an uploaded image
- `GET /api/v1/health` — Health check
- `GET /api/v1/metrics` — Model metrics
- (Add more as needed)

## Environment Variables
See `.env.example` for all supported variables (e.g., `ALLOWED_ORIGINS`).

## License
MIT License

