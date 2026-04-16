FROM python:3.11-slim

# Install ffmpeg (required for video/audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces runs as user 1000 — prepare writable dirs
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Writable directories for uploads/outputs inside the container
RUN mkdir -p uploads outputs && chown -R user:user /app

USER user

# HF Spaces expects port 7860
ENV FLASK_PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Single worker on purpose: job state lives in-process. Threads handle
# concurrent HTTP requests; the bounded ThreadPoolExecutor inside the app
# caps how many jobs run in parallel.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", \
     "--workers", "1", "--threads", "8", \
     "--timeout", "600", "--access-logfile", "-", "app:app"]
