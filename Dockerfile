# Production-ready Dockerfile for VeinWise-API-new (FastAPI)
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip uninstall -y numpy || true \
    && pip install numpy==1.26.4 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Use multiple workers for better concurrency
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"]
