# Dockerfile for River Level Prediction API
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY river_level_api.py .

# Create models directory
RUN mkdir -p models

# Copy model files ( need to add these)
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "river_level_api:app", "--host", "0.0.0.0", "--port", "8000"]