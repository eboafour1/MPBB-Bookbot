FROM python:3.9-slim

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

WORKDIR /app

# Now we’re inside /backend already
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
