# Base image
FROM docker.io/library/python:3.11-slim

# Working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r req.txt

# Retrain model when container builds
RUN python train.py

# Expose Streamlit port
EXPOSE 8501

# Launch app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
