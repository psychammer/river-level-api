# 1. Start with a lean, official Python base image.
# python:3.9-slim is a good choice for size and stability.
FROM python:3.9-slim

# 2. Set the working directory inside the container.
# This is where the code will live.
WORKDIR /app

# 3. Copy the requirements file first.
# This step is cached by Docker. If requirements.txt doesn't change,
# Docker won't re-install all the packages on every build, making it faster.
COPY requirements.txt .

# 4. Install the Python dependencies from requirements.txt.
# --no-cache-dir saves space in the final image.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy entire project's code into the container's working directory.
# This includes river_level_api.py, the 'models' folder, etc.
# The first '.' means "the current directory on computer".
# The second '.' means "the WORKDIR inside the container (/app)".
COPY . .

# 6. Expose the port that FastAPI application will run on.
# This tells Docker that the container will listen on port 8000.
EXPOSE 8000

# 7.  command to run application when the container starts.
# This tells uvicorn to run the 'app' object from the 'river_level_api.py' file.
# --host 0.0.0.0 is crucial for Docker, as it allows traffic from outside the container.
CMD ["uvicorn", "river_level_api:app", "--host", "0.0.0.0", "--port", "8000"]