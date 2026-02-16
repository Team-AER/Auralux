#!/usr/bin/env python3
"""Auralux local API server — ACE-Step MLX inference backend.

REST contract consumed by the Swift front-end:
- GET  /health
- POST /generate   (body: prompt, lyrics, tags, duration, variance, seed)
- GET  /jobs/<id>
- POST /jobs/<id>/cancel

The server lazily loads the ACE-Step model on first generation request and
runs inference on Apple Silicon via MLX.  If the model package is not yet
available it falls back to a silent-audio stub so the UI remains functional.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
import threading
import time
import traceback
import uuid
import wave
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auralux")

# ---------------------------------------------------------------------------
# Model loader (lazy singleton)
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_pipeline: Optional[Any] = None
_model_load_error: Optional[str] = None


def _get_pipeline() -> Any:
    """Return a loaded ACE-Step pipeline, or *None* if unavailable."""
    global _pipeline, _model_load_error
    if _pipeline is not None:
        return _pipeline
    with _model_lock:
        if _pipeline is not None:
            return _pipeline
        if _model_load_error is not None:
            return None
        try:
            log.info("Loading ACE-Step model (this may take a minute) …")
            from ace_step.pipeline import ACEStepPipeline

            model_dir = os.environ.get(
                "AURALUX_MODEL_DIR",
                str(
                    Path.home()
                    / "Library"
                    / "Application Support"
                    / "Auralux"
                    / "Models"
                ),
            )
            _pipeline = ACEStepPipeline(model_dir=model_dir)
            log.info("ACE-Step model loaded successfully.")
            return _pipeline
        except Exception as exc:
            _model_load_error = str(exc)
            log.warning("ACE-Step model unavailable: %s", exc)
            log.warning("Falling back to stub audio generation.")
            return None


# ---------------------------------------------------------------------------
# Job datastructures
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: float = 0.0
    message: Optional[str] = None
    audio_path: Optional[str] = None
    cancelled: bool = False


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)


JOBS = JobStore()


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------

def _write_silent_wav(path: Path, duration_sec: float, sample_rate: int = 44100) -> None:
    """Write a valid silent WAV file as a fallback when the model is unavailable."""
    num_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        silence = b"\x00\x00\x00\x00" * num_frames  # 2 channels × 16-bit
        wf.writeframes(silence)


def _run_job(
    job_id: str,
    prompt: str,
    lyrics: str,
    tags: List[str],
    duration: float,
    variance: float,
    seed: Optional[int],
) -> None:
    """Execute a generation job — real model or stub fallback."""
    JOBS.update(job_id, status="running", progress=0.05)

    output_dir = Path.home() / "Library" / "Application Support" / "Auralux" / "Generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_id}.wav"

    try:
        pipeline = _get_pipeline()

        if pipeline is not None:
            _run_real_inference(
                job_id=job_id,
                pipeline=pipeline,
                prompt=prompt,
                lyrics=lyrics,
                tags=tags,
                duration=duration,
                variance=variance,
                seed=seed,
                output_path=output_path,
            )
        else:
            _run_stub_inference(job_id, prompt, duration, output_path)

        JOBS.update(
            job_id,
            status="completed",
            progress=1.0,
            message=f"Generated track for: {prompt[:64]}",
            audio_path=str(output_path),
        )
    except Exception as exc:
        log.error("Generation failed for job %s: %s", job_id, exc)
        traceback.print_exc()
        JOBS.update(
            job_id,
            status="failed",
            message=f"Generation error: {exc}",
        )


def _run_real_inference(
    *,
    job_id: str,
    pipeline: Any,
    prompt: str,
    lyrics: str,
    tags: List[str],
    duration: float,
    variance: float,
    seed: Optional[int],
    output_path: Path,
) -> None:
    """Run ACE-Step MLX inference and write the output WAV."""
    import numpy as np
    import soundfile as sf

    full_prompt = prompt
    if tags:
        full_prompt = ", ".join(tags) + ". " + full_prompt

    JOBS.update(job_id, progress=0.10, message="Preparing generation …")

    def progress_callback(step: int, total_steps: int) -> bool:
        """Called by the pipeline each diffusion step. Return False to cancel."""
        job = JOBS.get(job_id)
        if job and job.cancelled:
            return False
        frac = 0.10 + 0.85 * (step / max(1, total_steps))
        JOBS.update(job_id, progress=min(0.95, frac), message=f"Step {step}/{total_steps}")
        return True

    generate_kwargs: Dict[str, Any] = {
        "prompt": full_prompt,
        "duration": duration,
    }
    if lyrics:
        generate_kwargs["lyrics"] = lyrics
    if seed is not None:
        generate_kwargs["seed"] = seed

    # The pipeline API may vary — adapt to the available signature
    if hasattr(pipeline, "generate"):
        audio = pipeline.generate(**generate_kwargs, callback=progress_callback)
    elif hasattr(pipeline, "__call__"):
        audio = pipeline(**generate_kwargs)
    else:
        raise RuntimeError("ACE-Step pipeline has no generate() or __call__ method")

    JOBS.update(job_id, progress=0.95, message="Saving audio …")

    # Handle various return types
    if isinstance(audio, dict):
        audio_data = audio.get("audio", audio.get("waveform"))
    elif isinstance(audio, tuple):
        audio_data = audio[0]
    else:
        audio_data = audio

    audio_np = np.asarray(audio_data, dtype=np.float32)
    if audio_np.ndim == 3:
        audio_np = audio_np.squeeze(0)
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)

    # audio_np shape: (channels, samples) — transpose for soundfile
    sf.write(str(output_path), audio_np.T, samplerate=44100, subtype="PCM_16")


def _run_stub_inference(
    job_id: str,
    prompt: str,
    duration: float,
    output_path: Path,
) -> None:
    """Simulate generation progress and write a valid silent WAV."""
    steps = 20
    for step in range(1, steps + 1):
        job = JOBS.get(job_id)
        if not job:
            return
        if job.cancelled:
            JOBS.update(job_id, status="cancelled", message="Job cancelled")
            return
        time.sleep(0.15)
        JOBS.update(
            job_id,
            progress=max(0.0, min(0.95, step / steps)),
            message=f"(stub) Step {step}/{steps}",
        )

    _write_silent_wav(output_path, duration_sec=duration)


class Handler(BaseHTTPRequestHandler):
    server_version = "AuraluxServer/0.1"

    def _read_json(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: Dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            model_loaded = _pipeline is not None
            self._send_json({
                "status": "ok",
                "modelLoaded": model_loaded,
                "modelError": _model_load_error,
            })
            return

        if self.path.startswith("/jobs/"):
            job_id = self.path.split("/")[2]
            job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(
                {
                    "jobID": job.id,
                    "status": job.status,
                    "progress": job.progress,
                    "message": job.message,
                    "audioPath": job.audio_path,
                }
            )
            return

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/generate":
            payload = self._read_json()
            prompt = str(payload.get("prompt", ""))
            lyrics = str(payload.get("lyrics", ""))
            tags = list(payload.get("tags", []))
            duration = float(payload.get("duration", 30))
            variance = float(payload.get("variance", 0.5))
            seed_raw = payload.get("seed")
            seed = int(seed_raw) if seed_raw is not None else None

            job = JOBS.create()
            worker = threading.Thread(
                target=_run_job,
                args=(job.id, prompt, lyrics, tags, duration, variance, seed),
                daemon=True,
            )
            worker.start()

            self._send_json({"jobID": job.id, "status": "queued", "message": "accepted"}, HTTPStatus.ACCEPTED)
            return

        if self.path.startswith("/jobs/") and self.path.endswith("/cancel"):
            job_id = self.path.split("/")[2]
            job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            JOBS.update(job_id, cancelled=True)
            self._send_json({"jobID": job.id, "status": "cancelling"})
            return

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        log.debug(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Auralux API server listening on http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
