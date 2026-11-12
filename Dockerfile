# Production Dockerfile for AI Generative Video Studio
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860"

# Install curl for healthcheck and ffmpeg for media validation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-privileged user and setup directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/output_videos && \
    chown -R appuser:appuser /app

# Install dependencies first for Docker caching
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and app files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser generate.py .
COPY --chown=appuser:appuser verify_backends.py .

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

ENTRYPOINT ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
