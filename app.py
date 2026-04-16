"""
StealthMask — Flask backend.
Applies imperceptible adversarial perturbation to images and video
to defeat AI vision models, OCR, and speech recognition.
"""

import logging
import os
import time
import uuid
import shutil
import zipfile
import subprocess
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    send_file, send_from_directory, abort
)
from scipy.fft import dctn, idctn
from scipy.interpolate import RectBivariateSpline
from scipy.signal import butter, lfilter, sosfilt
from dotenv import load_dotenv
import soundfile as sf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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

# How long uploads/outputs and their job records live before being swept.
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", str(60 * 60)))      # 1h default
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300"))  # 5m
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}

# In-memory job tracker: job_id -> {status, progress, files, created_at, finished_at}
jobs: dict = {}
jobs_lock = threading.Lock()

# Bounded worker pool — protects against OOM under load.
job_executor = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="stealthmask-job"
)

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

    # Thread-local RNG so parallel workers don't stomp on a shared seed.
    rng = np.random.default_rng(seed % (2**32)) if seed else np.random.default_rng()

    noise = np.zeros_like(img_array, dtype=np.float64)

    # --- Layer 1: Gaussian noise ---
    gaussian = rng.normal(0, epsilon * 0.45, img_array.shape)
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
    rng = np.random.default_rng(seed % (2**32))

    # Grid control points (low freq)
    grid_size = 5
    x = np.linspace(0, cols, grid_size)
    y = np.linspace(0, rows, grid_size)
    xv, yv = np.meshgrid(x, y)

    # Random displacement
    amp = 1.0 + strength * 0.8
    dx = rng.uniform(-amp, amp, xv.shape)
    dy = rng.uniform(-amp, amp, yv.shape)

    # Upscale displacement to original size
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

    rng = np.random.default_rng((seed + 77) % (2**32))
    amp = 1 + strength * 0.5

    # Apply noise to Chroma channels only (Cb, Cr)
    # Most AIs focus on Luma (Y) features
    noise_cr = rng.normal(0, amp, cr.shape).astype(np.int16)
    noise_cb = rng.normal(0, amp, cb.shape).astype(np.int16)

    cr = np.clip(cr.astype(np.int16) + noise_cr, 0, 255).astype(np.uint8)
    cb = np.clip(cb.astype(np.int16) + noise_cb, 0, 255).astype(np.uint8)

    yuv_p = cv2.merge([y, cr, cb])
    return cv2.cvtColor(yuv_p, cv2.COLOR_YCrCb2BGR)


_DCT_FREQ_MASK = np.array(
    [[1.0 if 2 <= i + j <= 6 else 0.0 for j in range(8)] for i in range(8)],
    dtype=np.float32,
)


def _dct_perturbation(img: np.ndarray, strength: int, seed: int) -> np.ndarray:
    """
    Attack DCT frequency domain — this is where pHash, content-ID,
    and most video fingerprinting systems operate.
    Injects noise into mid-frequency DCT coefficients per 8x8 block.
    Vectorized: dctn over the last two axes processes all blocks at once.
    """
    rng = np.random.default_rng((seed + 200) % (2**32))
    amp = 0.8 + strength * 0.4

    arr = img.astype(np.float32, copy=True)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    bh, bw = (h // 8) * 8, (w // 8) * 8
    if bh == 0 or bw == 0:
        return img

    # (bh/8, 8, bw/8, 8, C) -> (bh/8, bw/8, C, 8, 8)
    region = arr[:bh, :bw]
    blocks = region.reshape(bh // 8, 8, bw // 8, 8, c).transpose(0, 2, 4, 1, 3)
    blocks = np.ascontiguousarray(blocks)

    coeffs = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    noise = rng.uniform(-amp, amp, coeffs.shape).astype(np.float32) * _DCT_FREQ_MASK
    coeffs += noise
    blocks = idctn(coeffs, type=2, norm="ortho", axes=(-2, -1)).astype(np.float32)

    arr[:bh, :bw] = blocks.transpose(0, 3, 1, 4, 2).reshape(bh, bw, c)
    if img.ndim == 2:
        arr = arr[:, :, 0]
    return np.clip(arr, 0, 255).astype(np.uint8)


def _color_gamma_jitter(img: np.ndarray, strength: int, seed: int) -> np.ndarray:
    """
    Per-frame HSV micro-shift + gamma perturbation.
    Breaks temporal consistency that fingerprinting relies on.
    Each frame gets a slightly different color signature.
    """
    rng = np.random.default_rng((seed + 500) % (2**32))

    # Gamma perturbation (0.97 – 1.03, imperceptible)
    gamma = 1.0 + rng.uniform(-0.015, 0.015) * (strength / 5)
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)], dtype=np.uint8)
    result = cv2.LUT(img, lut)

    # HSV micro-shift
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue shift: ±1-3 degrees (out of 180 in OpenCV)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-1.5, 1.5) * (strength / 5)) % 180
    # Saturation shift: ±1-2%
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + rng.uniform(-0.015, 0.015) * (strength / 5)), 0, 255)
    # Value/brightness shift: ±0.5-1%
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + rng.uniform(-0.008, 0.008) * (strength / 5)), 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _break_audio_fingerprint(audio_path: str, output_path: str, strength: int, seed: int):
    """
    Attack audio fingerprinting systems (Shazam, AudioID, TikTok ACR).
    These systems match by finding spectral peaks (constellation map).
    Strategy: shift peaks, add phantom peaks, micro time-stretch.
    """
    data, sr = sf.read(audio_path, dtype="float64")
    rng = np.random.default_rng((seed + 999) % (2**32))
    n = data.shape[0]

    # --- 1. Micro time-stretch (imperceptible 0.3-1.5% change) ---
    # Resample to slightly different rate, then back
    stretch_factor = 1.0 + rng.uniform(-0.008, 0.008) * (strength / 5)
    new_n = int(n * stretch_factor)
    if data.ndim > 1:
        stretched = np.column_stack([
            np.interp(np.linspace(0, n - 1, new_n), np.arange(n), data[:, ch])
            for ch in range(data.shape[1])
        ])
    else:
        stretched = np.interp(np.linspace(0, n - 1, new_n), np.arange(n), data)

    # --- 2. Spectral peak injection (phantom peaks confuse constellation map) ---
    t = np.arange(new_n) / sr
    amp = 0.002 + (strength - 1) * 0.002  # 0.002 – 0.02
    # Inject tones at semi-random frequencies that create false spectral peaks
    phantom = np.zeros(new_n)
    for _ in range(3 + strength):
        freq = rng.uniform(300, 8000)
        phase = rng.uniform(0, 2 * np.pi)
        # Windowed burst (not constant — harder to filter out)
        burst_start = int(rng.integers(0, max(1, new_n - sr)))
        burst_len = int(rng.integers(sr // 8, sr // 2))
        burst_end = min(burst_start + burst_len, new_n)
        window = np.hanning(burst_end - burst_start)
        phantom[burst_start:burst_end] += np.sin(2 * np.pi * freq * t[burst_start:burst_end] + phase) * window * amp

    if stretched.ndim > 1:
        for ch in range(stretched.shape[1]):
            stretched[:, ch] += phantom
    else:
        stretched += phantom

    # --- 3. Micro pitch shift via harmonic distortion ---
    # Add very low-level harmonics that shift the spectral centroid
    for harmonic in [2, 3, 5]:
        h_amp = amp * 0.15 / harmonic
        freq_base = rng.uniform(100, 500)
        harm_tone = np.sin(2 * np.pi * freq_base * harmonic * t) * h_amp
        if stretched.ndim > 1:
            for ch in range(stretched.shape[1]):
                stretched[:, ch] += harm_tone
        else:
            stretched += harm_tone

    result = np.clip(stretched, -1.0, 1.0)
    sf.write(output_path, result, sr)


def process_image_file(src: Path, dst: Path, strength: int):
    """Load an image, apply the full TikTok Ads perturbation stack, save."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {src}")

    # Strip alpha — color-space ops (YCrCb, HSV, LUT) require 3 channels.
    # Alpha is dropped because all our perturbations target visual signal.
    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    seed = np.random.randint(0, 10000)

    img = _elastic_warp(img, strength, seed)
    img = _chroma_attack(img, strength, seed)
    img = _dct_perturbation(img, strength, seed)
    img = _color_gamma_jitter(img, strength, seed)

    processed = _adversarial_perturbation(
        img, strength, anti_ocr=True, distort_scene=True, seed=seed
    )

    # pHash breaker (zoom)
    h, w = processed.shape[:2]
    pad = int(min(h, w) * 0.006)
    if pad > 0:
        crop = processed[pad:h-pad, pad:w-pad]
        processed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)

    ext = dst.suffix.lower()
    if alpha is not None and ext in (".png", ".webp"):
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2BGRA)
        processed[:, :, 3] = alpha
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
    Mask amplitude is scaled to source RMS so quiet tracks don't get audible hiss.
    """
    data, sr = sf.read(audio_path, dtype="float64")
    mono = data.mean(axis=1) if data.ndim > 1 else data
    n = len(mono)
    rng = np.random.default_rng()

    # Scale mask to a fraction of source RMS — keeps it inaudible regardless
    # of source loudness. Floor at -60 dBFS so total silence still gets a tiny
    # mask (otherwise leading silence would be a fingerprint signature).
    src_rms = float(np.sqrt(np.mean(mono ** 2))) if n else 0.0
    src_rms = max(src_rms, 0.001)
    mask_ratio = 0.02 + (strength - 1) * 0.02   # 2% .. 20% of source RMS
    amp = src_rms * mask_ratio

    # --- Pink noise (1/f) ---
    white = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white)
    pink = pink / (np.max(np.abs(pink)) + 1e-9) * amp * 0.5

    # --- FM tones targeting Whisper mel-spectrogram bins ---
    t = np.arange(n) / sr
    fm_low = np.sin(2 * np.pi * (200 + 300 * np.sin(2 * np.pi * 0.5 * t)) * t) * amp * 0.3
    fm_high = np.sin(2 * np.pi * (3000 + 1500 * np.sin(2 * np.pi * 0.3 * t)) * t) * amp * 0.25

    # --- Phase distortion (band-limited noise in mid range) ---
    sos = butter(4, [1000, 4000], btype="bandpass", fs=sr, output="sos")
    phase_dist = sosfilt(sos, rng.standard_normal(n)) * amp * 0.2

    mask = pink + fm_low + fm_high + phase_dist

    if data.ndim > 1:
        # Slight per-channel decorrelation so stereo image isn't collapsed.
        ch = data.shape[1]
        result = data.copy()
        for i in range(ch):
            result[:, i] = data[:, i] + mask * (1.0 if i % 2 == 0 else 0.92)
    else:
        result = data + mask

    np.clip(result, -1.0, 1.0, out=result)
    sf.write(output_path, result, sr)


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------


# Cap workers per video so two concurrent video jobs don't oversubscribe
# the CPU. Each job still gets plenty of parallelism.
_VIDEO_WORKERS = max(2, (os.cpu_count() or 4) // max(1, MAX_CONCURRENT_JOBS))


def _process_video_frame(frame: np.ndarray, idx: int, strength: int,
                          global_seed: int, height: int, width: int,
                          pad_h: int, pad_w: int) -> tuple:
    """Run the full per-frame filter stack. Returns (raw BGR bytes, work_seconds)."""
    t0 = time.perf_counter()
    seed = global_seed + idx
    frame = _elastic_warp(frame, strength, global_seed + (idx // 2))
    frame = _chroma_attack(frame, strength, seed)
    frame = _dct_perturbation(frame, strength, seed)
    frame = _color_gamma_jitter(frame, strength, seed)

    processed = _adversarial_perturbation(
        frame, strength, anti_ocr=True, distort_scene=True, seed=seed
    )

    crop = processed[pad_h:height-pad_h, pad_w:width-pad_w]
    processed = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LANCZOS4)

    if not processed.flags["C_CONTIGUOUS"]:
        processed = np.ascontiguousarray(processed)
    return processed.tobytes(), time.perf_counter() - t0


def process_video_file(src: Path, dst: Path, strength: int, on_progress=None):
    """Process a video through the full TikTok Ads bypass pipeline."""
    tmp_dir = dst.parent / f"_tmp_{dst.stem}"
    tmp_dir.mkdir(exist_ok=True)

    try:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        global_seed = np.random.randint(0, 10000)
        raw_video = str(tmp_dir / "video_noaudio.mp4")

        # Randomize encoding parameters for a unique file signature.
        crf = str(np.random.randint(17, 20))
        gop = str(np.random.randint(24, 72))
        bf = str(np.random.randint(1, 4))
        tune = str(np.random.choice(["film", "animation", "grain"]))

        # Constant crop jitter derived from global_seed (was computed at idx=0
        # in the old sequential loop; hoisted so all workers see the same values).
        jitter = 0.003 + (global_seed % 50) * 0.0001
        pad_h, pad_w = int(height * jitter), int(width * jitter)

        mux_args = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast",
            "-tune", tune,
            "-crf", crf, "-pix_fmt", "yuv420p",
            "-g", gop, "-bf", bf,
            "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
            raw_video,
        ]

        ff = subprocess.Popen(
            mux_args, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        logger.info(
            "[video] start src=%s res=%dx%d fps=%.2f frames=%d "
            "cpu_count=%s workers=%d",
            src.name, width, height, fps, total_frames,
            os.cpu_count(), _VIDEO_WORKERS,
        )

        # Parallel frame processing: workers run the filter stack, main thread
        # drains futures in submission order and pipes raw bytes to ffmpeg.
        # Bounded inflight queue prevents the whole video from being read into RAM.
        max_inflight = _VIDEO_WORKERS * 3
        futures: deque = deque()
        written = 0
        total_work = 0.0        # sum of per-frame CPU seconds (from workers)
        wall_start = time.perf_counter()

        def _drain_one():
            nonlocal written, total_work
            data, work = futures.popleft().result()
            ff.stdin.write(data)
            total_work += work
            written += 1
            if on_progress and total_frames > 0:
                on_progress(int(written / total_frames * 90))

        try:
            with ThreadPoolExecutor(
                max_workers=_VIDEO_WORKERS,
                thread_name_prefix="stealthmask-frame",
            ) as pool:
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Block when the queue is full so we don't read ahead forever.
                    while len(futures) >= max_inflight:
                        _drain_one()

                    futures.append(pool.submit(
                        _process_video_frame, frame, idx, strength,
                        global_seed, height, width, pad_h, pad_w,
                    ))
                    idx += 1

                    # Opportunistic drain — head ready? pipe it without blocking encode.
                    while futures and futures[0].done():
                        _drain_one()

                # Drain remaining in order.
                while futures:
                    _drain_one()
        finally:
            cap.release()
            if ff.stdin:
                ff.stdin.close()
            ff.wait()
        if ff.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed with code {ff.returncode}")

        wall = time.perf_counter() - wall_start
        per_frame = (total_work / written) if written else 0.0
        # If parallelism is real, total_work >> wall (work happens concurrently).
        # effective_parallelism ~= number of cores actually busy on our work.
        eff_par = (total_work / wall) if wall > 0 else 0.0
        logger.info(
            "[video] done frames=%d wall=%.2fs work_sum=%.2fs "
            "per_frame=%.3fs encoded_fps=%.2f eff_parallelism=%.2fx (of %d workers)",
            written, wall, total_work, per_frame,
            written / wall if wall > 0 else 0.0,
            eff_par, _VIDEO_WORKERS,
        )

        # --- Audio ---
        audio_tmp = str(tmp_dir / "audio.wav")
        has_audio = _extract_audio(str(src), audio_tmp)

        if has_audio:
            masked = str(tmp_dir / "audio_mask.wav")
            _mask_audio(audio_tmp, masked, strength)

            fp_broken = str(tmp_dir / "audio_fp.wav")
            _break_audio_fingerprint(masked, fp_broken, strength, global_seed)

            stealth = str(tmp_dir / "audio_stealth.wav")
            pitch = 0.99 + (global_seed % 20) * 0.001
            filters = [
                "aphaser=in_gain=0.6:out_gain=0.8:delay=3:speed=1:type=t",
                "aecho=0.8:0.88:30:0.4",
                "tremolo=f=4:d=0.3",
                "vibrato=f=2:d=0.2",
                f"asetrate=44100*{pitch},aresample=44100",
            ]
            subprocess.run([
                "ffmpeg", "-y", "-i", fp_broken,
                "-af", ",".join(filters), stealth,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            _mux_video_audio(raw_video, stealth, str(dst))
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


def _run_job(job_id: str, files_meta: list, strength: int):
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
                process_image_file(src, dst, strength)
                _update_file_status(job_id, i, "done", 100)
            elif ext in VIDEO_EXTENSIONS:
                def progress_cb(pct, _i=i):
                    _update_file_status(job_id, _i, "processing", pct)
                process_video_file(src, dst, strength, on_progress=progress_cb)
                _update_file_status(job_id, i, "done", 100)
            else:
                _update_file_status(job_id, i, "error", 0)
        except Exception:
            logger.exception("Failed to process file %s", fm["original_name"])
            _update_file_status(job_id, i, "error", 0)

    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["finished_at"] = time.time()


def _update_file_status(job_id: str, file_idx: int, status: str, progress: int):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["files"][file_idx]["status"] = status
            jobs[job_id]["files"][file_idx]["progress"] = progress
            # Recalculate overall progress
            total = sum(f["progress"] for f in jobs[job_id]["files"])
            count = len(jobs[job_id]["files"])
            jobs[job_id]["progress"] = int(total / count) if count else 0


# ---------------------------------------------------------------------------
# TTL cleanup
# ---------------------------------------------------------------------------


def _sweep_expired_jobs(now: float | None = None):
    """Remove jobs older than JOB_TTL_SECONDS plus their on-disk artifacts."""
    now = now or time.time()
    expired = []
    with jobs_lock:
        for jid, job in list(jobs.items()):
            ts = job.get("finished_at") or job.get("created_at") or 0
            if now - ts > JOB_TTL_SECONDS:
                expired.append(jid)
                jobs.pop(jid, None)

    for jid in expired:
        for base in (UPLOAD_DIR, OUTPUT_DIR):
            shutil.rmtree(base / jid, ignore_errors=True)
    if expired:
        logger.info("Cleanup: removed %d expired job(s)", len(expired))

    # Sweep orphaned dirs (e.g. from a crash before job dict was populated).
    for base in (UPLOAD_DIR, OUTPUT_DIR):
        for entry in base.iterdir() if base.exists() else []:
            if not entry.is_dir():
                continue
            with jobs_lock:
                if entry.name in jobs:
                    continue
            try:
                if now - entry.stat().st_mtime > JOB_TTL_SECONDS:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                pass


def _cleanup_loop():
    while True:
        try:
            _sweep_expired_jobs()
        except Exception:
            logger.exception("Cleanup loop error")
        time.sleep(CLEANUP_INTERVAL_SECONDS)


threading.Thread(target=_cleanup_loop, name="stealthmask-cleanup", daemon=True).start()


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

    job_id = uuid.uuid4().hex
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
            "created_at": time.time(),
            "finished_at": None,
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
    strength = max(1, min(10, int(body.get("strength", 7))))

    job_executor.submit(_run_job, job_id, job["files"], strength)
    return jsonify({"status": "processing"})


@app.route("/api/status/<job_id>")
def status(job_id: str):
    """Return current processing status + per-file progress."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)


def _safe_filename(filename: str) -> str:
    """Strip path components to prevent directory traversal."""
    name = Path(filename).name
    if not name or name.startswith("."):
        abort(400)
    return name


@app.route("/api/preview/<job_id>/<filename>")
def preview(job_id: str, filename: str):
    """Serve an uploaded (original) file for before/after preview."""
    job_dir = UPLOAD_DIR / job_id
    return send_from_directory(str(job_dir), _safe_filename(filename))


@app.route("/api/result/<job_id>/<filename>")
def result(job_id: str, filename: str):
    """Serve a processed file."""
    job_dir = OUTPUT_DIR / job_id
    return send_from_directory(str(job_dir), _safe_filename(filename))


@app.route("/api/download/<job_id>/<filename>")
def download_single(job_id: str, filename: str):
    """Download a single processed file."""
    job_dir = OUTPUT_DIR / job_id
    return send_from_directory(str(job_dir), _safe_filename(filename), as_attachment=True)


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
    logger.info("StealthMask running at http://localhost:%s", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
