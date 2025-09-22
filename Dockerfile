# 1. Start with a lean, official Python base image.
FROM python:3.9-slim

# 2. Set the working directory inside the container.
WORKDIR /app

# 3. Copy requirements file first for caching.
COPY requirements.txt .

# 4. Install dependencies. This is a multi-step process for PyTorch Geometric.
# First, install PyTorch itself, forcing it to use the CPU-only version which is smaller.
RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Second, install the rest of the requirements. Pip will use the already-installed
# PyTorch and the find-links URL in the file to get the correct PyG packages.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your entire project's code and models into the container.
COPY . .

# 6. Expose the port the FastAPI application will run on.
EXPOSE 8000

# 7. The command to run your application when the container starts.
# CRITICAL CHANGE: This now points to 'prediction_api:app' instead of the old file.
CMD ["uvicorn", "prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]