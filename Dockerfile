FROM python:3.11-slim

# Install ffmpeg (required for video/audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
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

CMD ["python", "app.py"]
