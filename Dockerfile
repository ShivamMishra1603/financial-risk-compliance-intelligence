# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any)
# cmake might be needed if installing from source, but we try to use wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST to cache layer
COPY requirements.txt .

# Install Python dependencies
# We add fastapi uvicorn explicitly here or ensuring they are in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8000

# Metadata for the user/resume
LABEL maintainer="Shivam Mishra"
LABEL description="Financial Risk Intelligence Inference Service"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
