# Use official lightweight Python 3.12 slim base
FROM python:3.12-slim

# Set environment variables for non-interactive installs and unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    HF_HOME=/root/.cache/huggingface \
    PORT=8000

# Install minimal OS build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install pinned dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache Hugging Face and SentenceTransformers models during docker build
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import pipeline; \
print('Pre-caching sentence embeddings...'); \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Pre-caching summarization pipeline...'); \
pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', framework='pt'); \
print('Pre-caching QA pipeline...'); \
pipeline('question-answering', model='deepset/roberta-base-squad2', framework='pt'); \
print('Model pre-caching complete!')"

# Copy application source code and frontend
COPY app/ ./app/
COPY frontend/ ./frontend/

# Expose FastAPI listening port
EXPOSE 8000

# Health check to ensure Uvicorn is healthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]