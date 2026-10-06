# Use an official lightweight Python runtime
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for C/C++ extensions (ChromaDB/pydantic)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code into the container
COPY . .

# Hugging Face Spaces strictly requires port 7860
EXPOSE 7860

# Command to start your server (adjust app:app if your entry file is named differently, e.g. main:app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]