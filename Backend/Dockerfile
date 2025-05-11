# syntax=docker/dockerfile:1

# 1) Base image
FROM python:3.9-slim

# 2) Install system dependencies (including Calibre for ebook-convert)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      wget \
      ca-certificates \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      calibre && \
    rm -rf /var/lib/apt/lists/*

# Set working dir to inside backend/
WORKDIR /app

# Copy ONLY the backend files
COPY backend/ .  # <== this is key

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6) Expose your FastAPI port
EXPOSE 8000

# 7) Environment vars
#ENV PYTHONUNBUFFERED=1


# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]




