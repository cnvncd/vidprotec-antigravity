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
import atexit
import multiprocessing as mp
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    send_file, send_from_directory, abort
)
from scipy.fft import dctn, idctn
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
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

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
# Filters (TikTok-minimal bypass: elastic warp + DCT Y-noise + crop-resize
# for video; phantom peak injection + micro time-stretch for audio).
# Intensity is hardcoded; previous strength=7 is used throughout.
# ---------------------------------------------------------------------------


def _elastic_warp(img: np.ndarray, seed: int) -> np.ndarray:
    """Low-freq mesh distortion via a 5x5 random displacement grid upsampled
    with cv2.resize, applied with cv2.remap. Breaks pHash / SSIM."""
    rows, cols = img.shape[:2]
    rng = np.random.default_rng(seed % (2**32))

    grid_size = 5
    amp = 6.6  # equivalent to strength=7: 1.0 + 7 * 0.8
    dx = rng.uniform(-amp, amp, (grid_size, grid_size)).astype(np.float32)
    dy = rng.uniform(-amp, amp, (grid_size, grid_size)).astype(np.float32)
    dx_full = cv2.resize(dx, (cols, rows), interpolation=cv2.INTER_CUBIC)
    dy_full = cv2.resize(dy, (cols, rows), interpolation=cv2.INTER_CUBIC)

    base_x, base_y = np.meshgrid(
        np.arange(cols, dtype=np.float32),
        np.arange(rows, dtype=np.float32),
    )
    return cv2.remap(img, base_x + dx_full, base_y + dy_full,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


_DCT_FREQ_MASK = np.array(
    [[1.0 if 2 <= i + j <= 6 else 0.0 for j in range(8)] for i in range(8)],
    dtype=np.float32,
)


def _dct_perturbation(img: np.ndarray, seed: int) -> np.ndarray:
    """Blockwise 8x8 DCT on the Y (luminance) channel with mid-frequency
    coefficient noise. TikTok ACR / pHash / content-ID all fingerprint on Y."""
    rng = np.random.default_rng((seed + 200) % (2**32))
    amp = 3.6  # equivalent to strength=7: 0.8 + 7 * 0.4

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = yuv[:, :, 0]

    arr = y.astype(np.float32, copy=True)
    h, w = arr.shape
    bh, bw = (h // 8) * 8, (w // 8) * 8
    if bh == 0 or bw == 0:
        return img

    region = arr[:bh, :bw]
    blocks = region.reshape(bh // 8, 8, bw // 8, 8).transpose(0, 2, 1, 3)
    blocks = np.ascontiguousarray(blocks)

    coeffs = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    noise = rng.uniform(-amp, amp, coeffs.shape).astype(np.float32) * _DCT_FREQ_MASK
    coeffs += noise
    blocks = idctn(coeffs, type=2, norm="ortho", axes=(-2, -1)).astype(np.float32)

    arr[:bh, :bw] = blocks.transpose(0, 2, 1, 3).reshape(bh, bw)
    np.clip(arr, 0, 255, out=arr)
    yuv[:, :, 0] = arr.astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)


def _break_audio_fingerprint(audio_path: str, output_path: str, seed: int):
    """Break Shazam / AudioID / TikTok ACR via micro time-stretch + phantom
    spectral peak injection. These systems match on constellation maps of
    local spectral maxima; inserting false peaks and shifting timing moves
    the match below the similarity threshold."""
    data, sr = sf.read(audio_path, dtype="float64")
    rng = np.random.default_rng((seed + 999) % (2**32))
    n = data.shape[0]

    # Micro time-stretch (±1.1% — inaudible but shifts constellation timing).
    stretch_factor = 1.0 + rng.uniform(-0.011, 0.011)
    new_n = int(n * stretch_factor)
    if data.ndim > 1:
        stretched = np.column_stack([
            np.interp(np.linspace(0, n - 1, new_n), np.arange(n), data[:, ch])
            for ch in range(data.shape[1])
        ])
    else:
        stretched = np.interp(np.linspace(0, n - 1, new_n), np.arange(n), data)

    # Phantom spectral peaks: short windowed tone bursts at random freqs.
    t = np.arange(new_n) / sr
    amp = 0.014  # ~ -37 dBFS, inaudible against typical content
    phantom = np.zeros(new_n)
    for _ in range(10):
        freq = rng.uniform(300, 8000)
        phase = rng.uniform(0, 2 * np.pi)
        burst_start = int(rng.integers(0, max(1, new_n - sr)))
        burst_len = int(rng.integers(sr // 8, sr // 2))
        burst_end = min(burst_start + burst_len, new_n)
        window = np.hanning(burst_end - burst_start)
        phantom[burst_start:burst_end] += np.sin(
            2 * np.pi * freq * t[burst_start:burst_end] + phase
        ) * window * amp

    if stretched.ndim > 1:
        for ch in range(stretched.shape[1]):
            stretched[:, ch] += phantom
    else:
        stretched += phantom

    np.clip(stretched, -1.0, 1.0, out=stretched)
    sf.write(output_path, stretched, sr)


def process_image_file(src: Path, dst: Path):
    """Apply TikTok-bypass stack to an image: elastic warp + DCT Y-noise
    + tiny crop-resize (pHash boundary shift)."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {src}")

    # Strip alpha — YCrCb conversion needs 3 channels; re-attach at the end.
    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    seed = np.random.randint(0, 10000)
    img = _elastic_warp(img, seed)
    img = _dct_perturbation(img, seed)

    # pHash breaker: small zoom shifts frame boundaries.
    h, w = img.shape[:2]
    pad = int(min(h, w) * 0.006)
    if pad > 0:
        crop = img[pad:h-pad, pad:w-pad]
        img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    ext = dst.suffix.lower()
    if alpha is not None and ext in (".png", ".webp"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = alpha
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    elif ext == ".webp":
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_WEBP_QUALITY, 97])
    else:
        cv2.imwrite(str(dst), img)


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------


# Single-user mode: one video uses every CPU core. If MAX_CONCURRENT_JOBS
# is raised, reduce by that factor to avoid oversubscribing the host.
_VIDEO_WORKERS = max(2, (os.cpu_count() or 4) // max(1, MAX_CONCURRENT_JOBS))

# Frames per submitted task. Batching amortizes pickle + IPC overhead
# (~10–20 ms/frame) across multiple frames, so each worker spends more
# time on actual filter work and less waiting on bytes.
_FRAME_BATCH_SIZE = int(os.getenv("FRAME_BATCH_SIZE", "4"))

# Process pool for per-frame work. ProcessPool (not ThreadPool) because the
# filter stack hits GIL bottlenecks — Python bookkeeping between numpy/cv2
# calls serializes threads. Each process has its own GIL → true parallelism.
# Lazy-init: avoid forking workers at import time when gunicorn boots.
_frame_pool: ProcessPoolExecutor | None = None
_frame_pool_lock = threading.Lock()


def _get_frame_pool() -> ProcessPoolExecutor:
    global _frame_pool
    if _frame_pool is None:
        with _frame_pool_lock:
            if _frame_pool is None:
                # 'spawn' instead of default 'fork' — fork would make workers
                # inherit every open fd, including ffmpeg's stdin pipe, so
                # closing the pipe in the parent would never reach EOF and
                # ffmpeg would hang forever on wait().
                ctx = mp.get_context("spawn")
                _frame_pool = ProcessPoolExecutor(
                    max_workers=_VIDEO_WORKERS, mp_context=ctx,
                )
                logger.info("Initialized frame process pool with %d workers (spawn)",
                            _VIDEO_WORKERS)
    return _frame_pool


@atexit.register
def _shutdown_frame_pool():
    global _frame_pool
    if _frame_pool is not None:
        _frame_pool.shutdown(wait=False, cancel_futures=True)
        _frame_pool = None


def _process_video_batch(frames: list, start_idx: int,
                          global_seed: int, height: int, width: int,
                          pad_h: int, pad_w: int) -> tuple:
    """Run the minimal TikTok-bypass filter stack on a batch of frames.
    Returns (list of raw BGR bytes in order, dict of per-stage seconds)."""
    timings = {"warp": 0.0, "dct": 0.0, "crop": 0.0}
    out: list = []
    for offset, frame in enumerate(frames):
        idx = start_idx + offset
        seed = global_seed + idx

        t = time.perf_counter()
        frame = _elastic_warp(frame, global_seed + (idx // 2))
        timings["warp"] += time.perf_counter() - t

        t = time.perf_counter()
        frame = _dct_perturbation(frame, seed)
        timings["dct"] += time.perf_counter() - t

        t = time.perf_counter()
        crop = frame[pad_h:height-pad_h, pad_w:width-pad_w]
        processed = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        if not processed.flags["C_CONTIGUOUS"]:
            processed = np.ascontiguousarray(processed)
        out.append(processed.tobytes())
        timings["crop"] += time.perf_counter() - t
    return out, timings


def process_video_file(src: Path, dst: Path, on_progress=None):
    """Process a video through the TikTok-minimal bypass pipeline."""
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
            "cpu_count=%s workers=%d batch=%d",
            src.name, width, height, fps, total_frames,
            os.cpu_count(), _VIDEO_WORKERS, _FRAME_BATCH_SIZE,
        )

        # Parallel batched frame processing: workers handle batches of frames,
        # main thread drains batch futures in order and pipes raw bytes to
        # ffmpeg. Bounded inflight keeps memory bounded.
        max_inflight = _VIDEO_WORKERS * 3
        futures: deque = deque()
        written = 0
        total_timings = {"warp": 0.0, "dct": 0.0, "crop": 0.0}
        wall_start = time.perf_counter()

        def _drain_one():
            nonlocal written
            batch_bytes, timings = futures.popleft().result()
            for data in batch_bytes:
                ff.stdin.write(data)
                written += 1
            for k, v in timings.items():
                total_timings[k] += v
            if on_progress and total_frames > 0:
                on_progress(int(written / total_frames * 90))

        pool = _get_frame_pool()
        try:
            batch: list = []
            batch_start = 0
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                batch.append(frame)
                idx += 1

                if len(batch) >= _FRAME_BATCH_SIZE:
                    while len(futures) >= max_inflight:
                        _drain_one()
                    futures.append(pool.submit(
                        _process_video_batch, batch, batch_start,
                        global_seed, height, width, pad_h, pad_w,
                    ))
                    batch_start = idx
                    batch = []

                while futures and futures[0].done():
                    _drain_one()

            # Submit trailing partial batch.
            if batch:
                futures.append(pool.submit(
                    _process_video_batch, batch, batch_start,
                    global_seed, height, width, pad_h, pad_w,
                ))

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
        total_work = sum(total_timings.values())
        per_frame = (total_work / written) if written else 0.0
        eff_par = (total_work / wall) if wall > 0 else 0.0
        logger.info(
            "[video] frames_done frames=%d wall=%.2fs work_sum=%.2fs "
            "per_frame=%.3fs encoded_fps=%.2f eff_parallelism=%.2fx (of %d workers)",
            written, wall, total_work, per_frame,
            written / wall if wall > 0 else 0.0,
            eff_par, _VIDEO_WORKERS,
        )
        if written:
            per = {k: f"{(v / written) * 1000:.1f}ms" for k, v in total_timings.items()}
            logger.info("[video] per-filter per-frame: %s", per)

        # --- Audio ---
        logger.info("[video] extracting audio")
        audio_tmp = str(tmp_dir / "audio.wav")
        has_audio = _extract_audio(str(src), audio_tmp)

        if has_audio:
            logger.info("[video] breaking audio fingerprint")
            fp_broken = str(tmp_dir / "audio_fp.wav")
            _break_audio_fingerprint(audio_tmp, fp_broken, global_seed)

            logger.info("[video] applying pitch shift")
            stealth = str(tmp_dir / "audio_stealth.wav")
            pitch = 0.99 + (global_seed % 20) * 0.001
            _run_ffmpeg_safe([
                "ffmpeg", "-y", "-i", fp_broken,
                "-af", f"asetrate=44100*{pitch},aresample=44100",
                stealth,
            ], timeout=120, label="pitch shift")

            logger.info("[video] muxing video+audio")
            _mux_video_audio(raw_video, stealth, str(dst))
        else:
            logger.info("[video] no audio track, copying video only")
            shutil.copy2(raw_video, str(dst))

        if on_progress:
            on_progress(100)
        logger.info("[video] done src=%s", src.name)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_ffmpeg_safe(args: list, timeout: int, label: str):
    """Run ffmpeg with a timeout and capture stderr so we can see failures."""
    try:
        result = subprocess.run(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("[video] ffmpeg %s timed out after %ds", label, timeout)
        raise
    if result.returncode != 0:
        tail = (result.stderr or b"").decode(errors="replace")[-2000:]
        logger.error("[video] ffmpeg %s failed rc=%d stderr=%s",
                     label, result.returncode, tail)
        raise RuntimeError(f"ffmpeg {label} failed with code {result.returncode}")


def _extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio track from video using ffmpeg. Returns False if no audio."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        logger.error("[video] audio extract timed out")
        return False
    return result.returncode == 0 and os.path.exists(audio_path)


def _mux_video_audio(video_path: str, audio_path: str, output_path: str):
    """Combine processed video and audio into final file with total metadata strip."""
    _run_ffmpeg_safe([
        "ffmpeg", "-y",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
        "-shortest", output_path,
    ], timeout=120, label="mux")


# ---------------------------------------------------------------------------
# Job runner (background thread)
# ---------------------------------------------------------------------------


def _run_job(job_id: str, files_meta: list):
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
                process_image_file(src, dst)
                _update_file_status(job_id, i, "done", 100)
            elif ext in VIDEO_EXTENSIONS:
                def progress_cb(pct, _i=i):
                    _update_file_status(job_id, _i, "processing", pct)
                process_video_file(src, dst, on_progress=progress_cb)
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

    job_executor.submit(_run_job, job_id, job["files"])
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
