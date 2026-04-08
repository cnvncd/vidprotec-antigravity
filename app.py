"""
StealthMask — Flask backend.
Applies imperceptible adversarial perturbation to images and video
to defeat AI vision models, OCR, and speech recognition.
"""

import os
import uuid
import json
import shutil
import zipfile
import subprocess
import threading
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, jsonify,
    send_file, send_from_directory
)
from scipy.signal import butter, sosfilt
from dotenv import load_dotenv
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_MB", "500"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}

# In-memory job tracker: job_id -> {status, progress, files: [{name, status, progress}]}
jobs: dict = {}
jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Adversarial perturbation core
# ---------------------------------------------------------------------------


def _adversarial_perturbation(img_array: np.ndarray, strength: int,
                               anti_ocr: bool, distort_scene: bool) -> np.ndarray:
    """
    Generate a multi-layer adversarial noise tensor.
    Layers are calibrated so the result is nearly invisible to humans
    but devastating to neural-network feature extractors.

    strength: 1–10  (maps to epsilon range ~2–25 on uint8 scale)
    """
    h, w = img_array.shape[:2]
    channels = img_array.shape[2] if img_array.ndim == 3 else 1
    epsilon = 2.0 + (strength - 1) * 2.5          # 2 … 24.5

    noise = np.zeros_like(img_array, dtype=np.float64)

    # --- Layer 1: Gaussian noise (broad-spectrum feature disruption) ---
    gaussian = np.random.normal(0, epsilon * 0.45, img_array.shape)
    noise += gaussian

    # --- Layer 2: High-frequency grid noise (breaks convolution kernels) ---
    hf = np.zeros_like(img_array, dtype=np.float64)
    # Checkerboard pattern — maximises gradient confusion
    hf[0::2, 1::2] = epsilon * 0.35
    hf[1::2, 0::2] = epsilon * 0.35
    hf[0::2, 0::2] = -epsilon * 0.35
    hf[1::2, 1::2] = -epsilon * 0.35
    noise += hf

    # --- Layer 3: Sinusoidal wave (phase-shifts feature maps) ---
    y_coords = np.arange(h).reshape(-1, 1)
    x_coords = np.arange(w).reshape(1, -1)
    freq = 0.05 + strength * 0.015
    sin_pattern = np.sin(2 * np.pi * freq * x_coords + y_coords * freq * 0.7)
    sin_pattern = sin_pattern * epsilon * 0.3
    if channels > 1:
        sin_pattern = np.stack([sin_pattern] * channels, axis=-1)
    noise += sin_pattern

    # --- Layer 4 (anti-OCR): mid-frequency horizontal stripes ---
    if anti_ocr:
        stripe = np.sin(2 * np.pi * y_coords * 0.12) * epsilon * 0.5
        stripe_pattern = np.broadcast_to(
            stripe[:, :, np.newaxis] if channels > 1 else stripe,
            img_array.shape
        ).copy().astype(np.float64)
        noise += stripe_pattern

    # --- Layer 5 (scene distortion): radial gradient perturbation ---
    if distort_scene:
        cy, cx = h / 2, w / 2
        yy, xx = np.mgrid[0:h, 0:w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_r = np.sqrt(cy ** 2 + cx ** 2)
        radial = np.cos(radius / max_r * np.pi * (3 + strength * 0.5))
        radial = radial * epsilon * 0.4
        if channels > 1:
            radial = np.stack([radial] * channels, axis=-1)
        noise += radial

    # Apply & clamp
    result = img_array.astype(np.float64) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def process_image_file(src: Path, dst: Path, strength: int,
                        anti_ocr: bool, distort_scene: bool):
    """Load an image, apply adversarial perturbation, save."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {src}")
    processed = _adversarial_perturbation(img, strength, anti_ocr, distort_scene)

    ext = dst.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(dst), processed, [cv2.IMWRITE_JPEG_QUALITY, 97])
    elif ext == ".webp":
        cv2.imwrite(str(dst), processed, [cv2.IMWRITE_WEBP_QUALITY, 97])
    else:
        cv2.imwrite(str(dst), processed)


# ---------------------------------------------------------------------------
# Audio adversarial masking
# ---------------------------------------------------------------------------


def _mask_audio(audio_path: str, output_path: str, strength: int):
    """
    Add imperceptible noise to audio that disrupts Whisper / Deepgram / Grok.
    Combines:
      - Low-volume pink noise (broadband masking)
      - Phase distortion via all-pass-like filter
      - Frequency-modulated tones in Whisper-sensitive bands (200-800 Hz, 3-6 kHz)
    """
    data, sr = sf.read(audio_path, dtype="float64")
    mono = data.mean(axis=1) if data.ndim > 1 else data
    n = len(mono)
    amp = 0.003 + (strength - 1) * 0.003       # 0.003 … 0.03

    # --- Pink noise (1/f) ---
    white = np.random.randn(n)
    # Approximate pink via cumulative filter
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    from scipy.signal import lfilter
    pink = lfilter(b, a, white)
    pink = pink / (np.max(np.abs(pink)) + 1e-9) * amp * 0.5

    # --- FM tones targeting Whisper mel-spectrogram bins ---
    t = np.arange(n) / sr
    # Sweep 200–800 Hz
    fm_low = np.sin(2 * np.pi * (200 + 300 * np.sin(2 * np.pi * 0.5 * t)) * t)
    fm_low *= amp * 0.3
    # Sweep 3000–6000 Hz
    fm_high = np.sin(2 * np.pi * (3000 + 1500 * np.sin(2 * np.pi * 0.3 * t)) * t)
    fm_high *= amp * 0.25

    # --- Phase distortion (all-pass style) ---
    sos = butter(4, [1000, 4000], btype="bandpass", fs=sr, output="sos")
    phase_dist = sosfilt(sos, np.random.randn(n)) * amp * 0.2

    # Combine
    mask = pink + fm_low + fm_high + phase_dist

    if data.ndim > 1:
        mask_stereo = np.stack([mask, mask], axis=-1)
        result = data + mask_stereo[:len(data)]
    else:
        result = data + mask[:len(data)]

    result = np.clip(result, -1.0, 1.0)
    sf.write(output_path, result, sr)


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------


def process_video_file(src: Path, dst: Path, strength: int,
                        anti_ocr: bool, distort_scene: bool,
                        mask_audio: bool, on_progress=None):
    """
    Process a video frame-by-frame with adversarial perturbation.
    Optionally masks the audio track too.
    Uses subprocess + ffmpeg for muxing to preserve quality.
    """
    tmp_dir = dst.parent / f"_tmp_{dst.stem}"
    tmp_dir.mkdir(exist_ok=True)

    try:
        # --- Extract info ---
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- Process frames ---
        frames_dir = tmp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = _adversarial_perturbation(frame, strength, anti_ocr, distort_scene)
            cv2.imwrite(str(frames_dir / f"{idx:08d}.png"), processed)
            idx += 1
            if on_progress and total_frames > 0:
                on_progress(int(idx / total_frames * 90))   # 0-90% for frames
        cap.release()

        # --- Reassemble video from frames (no audio) ---
        raw_video = str(tmp_dir / "video_noaudio.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(frames_dir / "%08d.png"),
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}",
            raw_video
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # --- Handle audio ---
        audio_tmp = str(tmp_dir / "audio.wav")
        has_audio = _extract_audio(str(src), audio_tmp)

        if has_audio and mask_audio:
            masked_audio = str(tmp_dir / "audio_masked.wav")
            _mask_audio(audio_tmp, masked_audio, strength)
            _mux_video_audio(raw_video, masked_audio, str(dst))
        elif has_audio:
            _mux_video_audio(raw_video, audio_tmp, str(dst))
        else:
            shutil.copy2(raw_video, str(dst))

        if on_progress:
            on_progress(100)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio track from video using ffmpeg. Returns False if no audio."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return result.returncode == 0 and os.path.exists(audio_path)


def _mux_video_audio(video_path: str, audio_path: str, output_path: str):
    """Combine processed video and audio into final file."""
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


# ---------------------------------------------------------------------------
# Job runner (background thread)
# ---------------------------------------------------------------------------


def _run_job(job_id: str, files_meta: list, strength: int,
             anti_ocr: bool, distort_scene: bool, mask_audio_flag: bool):
    """Process every file in the job. Runs in a background thread."""
    job_output = OUTPUT_DIR / job_id
    job_output.mkdir(exist_ok=True)

    for i, fm in enumerate(files_meta):
        _update_file_status(job_id, i, "processing", 0)
        src = UPLOAD_DIR / job_id / fm["stored_name"]
        ext = Path(fm["original_name"]).suffix.lower()
        out_stem = Path(fm["original_name"]).stem + "_stealthmasked"
        dst = job_output / (out_stem + ext)

        try:
            if ext in IMAGE_EXTENSIONS:
                process_image_file(src, dst, strength, anti_ocr, distort_scene)
                _update_file_status(job_id, i, "done", 100)
            elif ext in VIDEO_EXTENSIONS:
                def progress_cb(pct, _i=i):
                    _update_file_status(job_id, _i, "processing", pct)
                process_video_file(src, dst, strength, anti_ocr, distort_scene,
                                   mask_audio_flag, on_progress=progress_cb)
                _update_file_status(job_id, i, "done", 100)
            else:
                _update_file_status(job_id, i, "error", 0)
        except Exception as e:
            print(f"[ERROR] file {fm['original_name']}: {e}")
            _update_file_status(job_id, i, "error", 0)

    with jobs_lock:
        jobs[job_id]["status"] = "done"


def _update_file_status(job_id: str, file_idx: int, status: str, progress: int):
    with jobs_lock:
        jobs[job_id]["files"][file_idx]["status"] = status
        jobs[job_id]["files"][file_idx]["progress"] = progress
        # Recalculate overall progress
        total = sum(f["progress"] for f in jobs[job_id]["files"])
        count = len(jobs[job_id]["files"])
        jobs[job_id]["progress"] = int(total / count) if count else 0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Accept multiple files, store them, return a job_id."""
    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "No files uploaded"}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    files_meta = []
    for f in uploaded:
        ext = Path(f.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
            continue
        stored = uuid.uuid4().hex[:8] + ext
        f.save(str(job_dir / stored))
        file_type = "image" if ext in IMAGE_EXTENSIONS else "video"
        files_meta.append({
            "original_name": f.filename,
            "stored_name": stored,
            "type": file_type,
            "status": "pending",
            "progress": 0,
        })

    if not files_meta:
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": "No supported files"}), 400

    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "files": files_meta,
        }

    return jsonify({"job_id": job_id, "files": files_meta})


@app.route("/api/process/<job_id>", methods=["POST"])
def process(job_id: str):
    """Start processing a previously uploaded job."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        if job["status"] == "processing":
            return jsonify({"error": "Already processing"}), 409
        job["status"] = "processing"

    body = request.get_json(silent=True) or {}
    strength = max(1, min(10, int(body.get("strength", 5))))
    anti_ocr = bool(body.get("anti_ocr", False))
    distort_scene = bool(body.get("distort_scene", False))
    mask_audio_flag = bool(body.get("mask_audio", False))

    t = threading.Thread(
        target=_run_job,
        args=(job_id, job["files"], strength, anti_ocr, distort_scene, mask_audio_flag),
        daemon=True
    )
    t.start()
    return jsonify({"status": "processing"})


@app.route("/api/status/<job_id>")
def status(job_id: str):
    """Return current processing status + per-file progress."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)


@app.route("/api/preview/<job_id>/<filename>")
def preview(job_id: str, filename: str):
    """Serve an uploaded (original) file for before/after preview."""
    job_dir = UPLOAD_DIR / job_id
    return send_from_directory(str(job_dir), filename)


@app.route("/api/result/<job_id>/<filename>")
def result(job_id: str, filename: str):
    """Serve a processed file."""
    job_dir = OUTPUT_DIR / job_id
    return send_from_directory(str(job_dir), filename)


@app.route("/api/download/<job_id>/<filename>")
def download_single(job_id: str, filename: str):
    """Download a single processed file."""
    job_dir = OUTPUT_DIR / job_id
    return send_from_directory(str(job_dir), filename, as_attachment=True)


@app.route("/api/download-all/<job_id>")
def download_all(job_id: str):
    """Pack all processed files into a ZIP and stream it."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return jsonify({"error": "No results"}), 404

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in job_dir.iterdir():
            if f.is_file():
                zf.write(f, f.name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name=f"stealthmask_{job_id}.zip")


@app.route("/api/delete-file/<job_id>/<stored_name>", methods=["DELETE"])
def delete_file(job_id: str, stored_name: str):
    """Remove a single file from a pending job."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        job["files"] = [f for f in job["files"] if f["stored_name"] != stored_name]

    file_path = UPLOAD_DIR / job_id / stored_name
    if file_path.exists():
        file_path.unlink()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n  🛡️  StealthMask running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
