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
                               anti_ocr: bool, distort_scene: bool,
                               seed: int = 0) -> np.ndarray:
    """
    Generate a multi-layer adversarial noise tensor.
    seed: if provided, shifts the phase of the patterns (for video dynamic noise).
    """
    h, w = img_array.shape[:2]
    channels = img_array.shape[2] if img_array.ndim == 3 else 1
    epsilon = 2.0 + (strength - 1) * 2.5

    # Random patterns grounded by seed
    if seed:
        np.random.seed(seed % (2**32))

    noise = np.zeros_like(img_array, dtype=np.float64)

    # --- Layer 1: Gaussian noise ---
    gaussian = np.random.normal(0, epsilon * 0.45, img_array.shape)
    noise += gaussian

    # --- Layer 2: High-frequency grid noise ---
    hf = np.zeros_like(img_array, dtype=np.float64)
    hf[0::2, 1::2] = epsilon * 0.35
    hf[1::2, 0::2] = epsilon * 0.35
    hf[0::2, 0::2] = -epsilon * 0.35
    hf[1::2, 1::2] = -epsilon * 0.35
    noise += hf

    # --- Layer 3: Sinusoidal wave (with phase shift) ---
    phase_x = (seed * 0.13) if seed else 0
    phase_y = (seed * 0.07) if seed else 0
    y_coords = np.arange(h).reshape(-1, 1)
    x_coords = np.arange(w).reshape(1, -1)
    freq = 0.05 + strength * 0.015
    sin_pattern = np.sin(2 * np.pi * freq * x_coords + phase_x + y_coords * freq * 0.7 + phase_y)
    sin_pattern = sin_pattern * epsilon * 0.3
    if channels > 1:
        sin_pattern = np.stack([sin_pattern] * channels, axis=-1)
    noise += sin_pattern

    # --- Layer 4 (anti-OCR): mid-frequency horizontal stripes ---
    if anti_ocr:
        stripe = np.sin(2 * np.pi * y_coords * 0.12 + phase_y * 0.5) * epsilon * 0.5
        stripe_pattern = np.broadcast_to(
            stripe[:, :, np.newaxis] if channels > 1 else stripe,
            img_array.shape
        ).copy().astype(np.float64)
        noise += stripe_pattern

    # --- Layer 5 (scene distortion): radial gradient perturbation ---
    if distort_scene:
        cy, cx = h / 2 + (seed % 10 - 5 if seed else 0), w / 2 + (seed % 14 - 7 if seed else 0)
        yy, xx = np.mgrid[0:h, 0:w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_r = np.sqrt((h/2) ** 2 + (w/2) ** 2)
        radial = np.cos(radius / max_r * np.pi * (3 + strength * 0.5) + phase_x * 0.2)
        radial = radial * epsilon * 0.4
        if channels > 1:
            radial = np.stack([radial] * channels, axis=-1)
        noise += radial

    # Apply & clamp
    result = img_array.astype(np.float64) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def _elastic_warp(img: np.ndarray, strength: int, seed: int) -> np.ndarray:
    """
    Apply low-frequency mesh distortion to break pHash and SSIM.
    strength: 1-10
    seed: random factor
    """
    rows, cols = img.shape[:2]
    np.random.seed(seed % (2**32))

    # Grid control points (low freq)
    grid_size = 5
    x = np.linspace(0, cols, grid_size)
    y = np.linspace(0, rows, grid_size)
    xv, yv = np.meshgrid(x, y)

    # Random displacement
    amp = 1.0 + strength * 0.8
    dx = np.random.uniform(-amp, amp, xv.shape)
    dy = np.random.uniform(-amp, amp, yv.shape)

    # Upscale displacement to original size
    from scipy.interpolate import RectBivariateSpline
    fx = RectBivariateSpline(y, x, dx)
    fy = RectBivariateSpline(y, x, dy)

    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = map_x.astype(np.float32) + fx(np.arange(rows), np.arange(cols)).astype(np.float32)
    map_y = map_y.astype(np.float32) + fy(np.arange(rows), np.arange(cols)).astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def _chroma_attack(img: np.ndarray, strength: int, seed: int) -> np.ndarray:
    """
    Attack YCbCr color space specifically.
    """
    # Convert to YUV (YCbCr)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(yuv)

    np.random.seed((seed + 77) % (2**32))
    amp = 1 + strength * 0.5

    # Apply noise to Chroma channels only (Cb, Cr)
    # Most AIs focus on Luma (Y) features
    noise_cr = np.random.normal(0, amp, cr.shape).astype(np.int16)
    noise_cb = np.random.normal(0, amp, cb.shape).astype(np.int16)

    cr = np.clip(cr.astype(np.int16) + noise_cr, 0, 255).astype(np.uint8)
    cb = np.clip(cb.astype(np.int16) + noise_cb, 0, 255).astype(np.uint8)

    yuv_p = cv2.merge([y, cr, cb])
    return cv2.cvtColor(yuv_p, cv2.COLOR_YCrCb2BGR)


def process_image_file(src: Path, dst: Path, strength: int, profile: str, custom_flags: dict = None):
    """Load an image, apply adversarial perturbation based on profile, save."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {src}")

    # Default profile settings
    settings = {
        "warp": profile in ("tt_ads", "ghost"),
        "chroma": profile in ("tt_ads", "ghost"),
        "anti_ocr": profile in ("tt_ads",),
        "distort_scene": profile in ("tt_ads", "ghost"),
        "v4_semantic": profile in ("tt_ads", "invisible"),
        "deep_stealth": profile in ("tt_ads", "ghost")
    }
    # Override with custom flags if provided
    if custom_flags:
        settings.update(custom_flags)

    seed = np.random.randint(0, 10000)

    # v3 Neural Warp
    if settings["warp"]:
        img = _elastic_warp(img, strength, seed)
    
    # v4 Chroma Attack
    if settings["chroma"]:
        img = _chroma_attack(img, strength, seed)

    processed = _adversarial_perturbation(
        img, strength, 
        settings["anti_ocr"], 
        settings["distort_scene"], 
        seed=seed
    )

    # v2/v4 pHash breaker (zoom)
    if settings["deep_stealth"] or settings["v4_semantic"]:
        h, w = processed.shape[:2]
        pad = int(min(h, w) * 0.006)
        if pad > 0:
            crop = processed[pad:h-pad, pad:w-pad]
            processed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)

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


def process_video_file(src: Path, dst: Path, strength: int, profile: str, 
                        custom_flags: dict = None, on_progress=None):
    """
    Process a video using smart profiles. 
    v5.0 Unified Engine.
    """
    tmp_dir = dst.parent / f"_tmp_{dst.stem}"
    tmp_dir.mkdir(exist_ok=True)

    # Default profile settings
    settings = {
        "warp": profile in ("tt_ads", "ghost"),
        "chroma": profile in ("tt_ads", "ghost"),
        "anti_ocr": profile in ("tt_ads",),
        "distort_scene": profile in ("tt_ads", "ghost"),
        "mask_audio": profile in ("tt_ads", "ghost"),
        "audio_stealth": profile in ("tt_ads",),
        "v4_semantic": profile in ("tt_ads", "invisible"),
        "deep_stealth": profile in ("tt_ads", "ghost"),
        "metadata_strip": True
    }
    if custom_flags:
        settings.update(custom_flags)

    try:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_dir = tmp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        idx = 0
        global_seed = np.random.randint(0, 10000)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            seed = global_seed + idx
            
            if settings["warp"]:
                frame = _elastic_warp(frame, strength, global_seed + (idx // 2))
            
            if settings["chroma"]:
                frame = _chroma_attack(frame, strength, seed)

            processed = _adversarial_perturbation(
                frame, strength, 
                settings["anti_ocr"], 
                settings["distort_scene"], 
                seed=seed
            )

            if settings["deep_stealth"] or settings["v4_semantic"]:
                if idx == 0:
                    jitter = 0.003 + (global_seed % 50) * 0.0001
                    pad_h, pad_w = int(height * jitter), int(width * jitter)
                crop = processed[pad_h:height-pad_h, pad_w:width-pad_w]
                processed = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LANCZOS4)

            cv2.imwrite(str(frames_dir / f"{idx:08d}.png"), processed)
            idx += 1
            if on_progress and total_frames > 0:
                on_progress(int(idx / total_frames * 90))
        cap.release()

        # --- Reassemble ---
        raw_video = str(tmp_dir / "video_noaudio.mp4")
        mux_args = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(frames_dir / "%08d.png"),
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}"
        ]
        if settings["metadata_strip"]:
            mux_args += ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact"]
        
        mux_args.append(raw_video)
        subprocess.run(mux_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # --- Audio ---
        audio_tmp = str(tmp_dir / "audio.wav")
        has_audio = _extract_audio(str(src), audio_tmp)

        if has_audio:
            final_audio = audio_tmp
            if settings["mask_audio"]:
                masked = str(tmp_dir / "audio_mask.wav")
                _mask_audio(audio_tmp, masked, strength)
                final_audio = masked

            if settings["audio_stealth"]:
                stealth = str(tmp_dir / "audio_stealth.wav")
                filters = [
                    f"aphaser=in_gain=0.6:out_gain=0.8:delay=3:speed=1:type=t",
                    f"aecho=0.8:0.88:30:0.4",
                    f"tremolo=f=4:d=0.3",
                    f"vibrato=f=2:d=0.2"
                ]
                # Pitch jitter
                pitch = 0.99 + (global_seed % 20) * 0.001
                filters.append(f"asetrate=44100*{pitch},aresample=44100")
                
                subprocess.run([
                    "ffmpeg", "-y", "-i", final_audio, 
                    "-af", ",".join(filters), stealth
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                final_audio = stealth

            _mux_video_audio(raw_video, final_audio, str(dst))
        else:
            shutil.copy2(raw_video, str(dst))

        if on_progress: on_progress(100)
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
    """Combine processed video and audio into final file with total metadata strip."""
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
        "-shortest", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


# ---------------------------------------------------------------------------
# Job runner (background thread)
# ---------------------------------------------------------------------------


def _run_job_v5(job_id: str, files_meta: list, strength: int, profile: str, custom_flags: dict):
    """Process every file in the job. Runs in a background thread."""
    job_output = OUTPUT_DIR / job_id
    job_output.mkdir(exist_ok=True)

    for i, fm in enumerate(files_meta):
        _update_file_status(job_id, i, "processing", 0)
        src = UPLOAD_DIR / job_id / fm["stored_name"]
        ext = Path(fm["original_name"]).suffix.lower()
        dst = job_output / (Path(fm["original_name"]).stem + "_stealthmasked" + ext)

        try:
            if ext in IMAGE_EXTENSIONS:
                process_image_file(src, dst, strength, profile, custom_flags)
                _update_file_status(job_id, i, "done", 100)
            elif ext in VIDEO_EXTENSIONS:
                def progress_cb(pct, _i=i):
                    _update_file_status(job_id, _i, "processing", pct)
                process_video_file(src, dst, strength, profile, custom_flags, on_progress=progress_cb)
                _update_file_status(job_id, i, "done", 100)
            else:
                _update_file_status(job_id, i, "error", 0)
        except Exception as e:
            print(f"[ERROR] file {fm['original_name']}: {e}")
            _update_file_status(job_id, i, "error", 0)

    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = "done"

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
    strength = max(1, min(10, int(body.get("strength", 7))))
    profile = body.get("profile", "tt_ads")
    custom_flags = body.get("custom_flags", {})

    t = threading.Thread(
        target=_run_job_v5,
        args=(job_id, job["files"], strength, profile, custom_flags),
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
