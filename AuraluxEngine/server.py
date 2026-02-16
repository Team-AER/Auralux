#!/usr/bin/env python3
"""Auralux local API server — ACE-Step v1.5 inference backend.

Thin adapter that wraps ACE-Step 1.5's Python inference API while
keeping the REST contract the Swift front-end already speaks:

  GET  /health
  POST /generate       {prompt, lyrics, tags, duration, seed, ...}
  GET  /jobs/<id>
  POST /jobs/<id>/cancel
  POST /models/download

The DiT runs via PyTorch MPS on Apple Silicon.  The optional 5Hz LM
uses the MLX backend for native acceleration (set ACESTEP_LM_BACKEND=mlx).

Models are auto-downloaded from HuggingFace on first generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
import wave
from dataclasses import dataclass
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

# Ensure MPS fallback is enabled before any PyTorch import.
# This prevents Metal shader assertion crashes for *unsupported* MPS ops
# by transparently routing them to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ---------------------------------------------------------------------------
# Monkey-patch: masked_fill on MPS
# ---------------------------------------------------------------------------
# PyTorch's masked_fill is technically "supported" on MPS so the fallback env
# var above does NOT help.  However the underlying Metal shader
# (masked_fill_scalar_strided_32bit) has a buffer-binding bug that triggers:
#
#   "Read-only bytes are being bound at index 6 to a shader argument
#    with write access enabled"
#
# This kills the process.  The workaround is to move tensors to CPU for
# the masked_fill call and copy the result back.
try:
    import torch as _torch

    _orig_masked_fill = _torch.Tensor.masked_fill
    _orig_masked_fill_ = _torch.Tensor.masked_fill_

    def _safe_masked_fill(self, mask, value):
        if self.device.type == "mps":
            cpu_mask = mask.cpu() if isinstance(mask, _torch.Tensor) else mask
            return _orig_masked_fill(self.cpu(), cpu_mask, value).to(self.device)
        return _orig_masked_fill(self, mask, value)

    def _safe_masked_fill_(self, mask, value):
        if self.device.type == "mps":
            cpu_mask = mask.cpu() if isinstance(mask, _torch.Tensor) else mask
            result = _orig_masked_fill(self.cpu(), cpu_mask, value).to(self.device)
            self.data.copy_(result.data)
            return self
        return _orig_masked_fill_(self, mask, value)

    _torch.Tensor.masked_fill = _safe_masked_fill
    _torch.Tensor.masked_fill_ = _safe_masked_fill_
    log.info("Patched torch.Tensor.masked_fill for MPS safety")
except ImportError:
    pass  # torch not yet available; patch will be skipped


# ---------------------------------------------------------------------------
# Monkey-patch: audio-code decoding on MPS
# ---------------------------------------------------------------------------
# The MPS Metal shader ``mul_dense_scalar_float_float`` has a buffer-binding
# bug that triggers a fatal assertion inside torch.inference_mode():
#
#   "Read-only bytes are being bound at index 2 to a shader argument
#    with write access enabled"
#
# This kills the server process during _decode_audio_codes_to_latents (the
# quantizer / detokenizer forward pass).  Unlike masked_fill, scalar mul IS
# reported as "supported" on MPS so PYTORCH_ENABLE_MPS_FALLBACK does not
# help.
#
# The fix temporarily moves the small quantizer and detokenizer modules to
# CPU for the decode, then restores them to MPS.  Since the heavy DiT and
# VAE inference runs entirely on MLX, the performance impact is negligible.

def _patch_mps_audio_code_decode() -> None:
    """Replace _decode_audio_codes_to_latents with a CPU-safe variant."""
    try:
        import torch
        from acestep.core.generation.handler.audio_codes import AudioCodesMixin
    except ImportError:
        return

    _orig_decode = AudioCodesMixin._decode_audio_codes_to_latents

    def _cpu_safe_decode(self, code_str):  # type: ignore[override]
        if not hasattr(self, "device") or str(self.device).split(":")[0] != "mps":
            return _orig_decode(self, code_str)

        if (
            self.model is None
            or not hasattr(self.model, "tokenizer")
            or not hasattr(self.model, "detokenizer")
        ):
            return None

        code_ids = self._parse_audio_code_string(code_str)
        if not code_ids:
            return None

        mps_dev = self.device

        with self._load_model_context("model"):
            quantizer = self.model.tokenizer.quantizer
            detokenizer = self.model.detokenizer

            quantizer.cpu()
            detokenizer.cpu()
            try:
                indices = torch.tensor(code_ids, device="cpu", dtype=torch.long)
                indices = indices.unsqueeze(0).unsqueeze(-1)

                quantized = quantizer.get_output_from_indices(indices)
                if quantized.dtype != self.dtype:
                    quantized = quantized.to(self.dtype)
                lm_hints = detokenizer(quantized)
                return lm_hints.to(mps_dev)
            finally:
                quantizer.to(mps_dev)
                detokenizer.to(mps_dev)

    AudioCodesMixin._decode_audio_codes_to_latents = _cpu_safe_decode
    log.info("Patched _decode_audio_codes_to_latents for MPS safety (CPU fallback)")


# ---------------------------------------------------------------------------
# Paths – resolve the cloned ACE-Step 1.5 repo so we can import `acestep`
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ACE_STEP_DIR = SCRIPT_DIR / "ACE-Step-1.5"
CHECKPOINTS_DIR = ACE_STEP_DIR / "checkpoints"

if ACE_STEP_DIR.is_dir():
    sys.path.insert(0, str(ACE_STEP_DIR))

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
_dit_handler: Optional[Any] = None
_llm_handler: Optional[Any] = None
_init_error: Optional[str] = None
_init_done = False


def _ensure_initialized() -> bool:
    """Lazily initialize the ACE-Step handlers.  Returns True on success."""
    global _dit_handler, _llm_handler, _init_error, _init_done

    if _init_done:
        return _dit_handler is not None

    with _init_lock:
        if _init_done:
            return _dit_handler is not None

        try:
            log.info("Importing ACE-Step 1.5 modules …")
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler

            _patch_mps_audio_code_decode()

            project_root = str(ACE_STEP_DIR)
            os.makedirs(str(CHECKPOINTS_DIR), exist_ok=True)

            # --- DiT handler ---
            dit = AceStepHandler()
            config_path = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
            device = os.environ.get("ACESTEP_DEVICE", "auto")
            offload = os.environ.get("ACESTEP_OFFLOAD_TO_CPU", "false").lower() in ("1", "true", "yes")

            log.info("Initializing DiT handler (config=%s, device=%s) …", config_path, device)
            status_msg, ok = dit.initialize_service(
                project_root=project_root,
                config_path=config_path,
                device=device,
                offload_to_cpu=offload,
            )
            if not ok:
                raise RuntimeError(f"DiT init failed: {status_msg}")
            log.info("DiT handler ready: %s", status_msg)
            _dit_handler = dit

            # --- LLM handler (optional, best-effort) ---
            init_llm = os.environ.get("ACESTEP_INIT_LLM", "auto").lower()
            if init_llm == "false":
                log.info("LLM disabled via ACESTEP_INIT_LLM=false")
                _llm_handler = LLMHandler()
            else:
                llm = LLMHandler()
                lm_model = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
                lm_backend = os.environ.get("ACESTEP_LM_BACKEND", "mlx")
                lm_device = os.environ.get("ACESTEP_LM_DEVICE", device)
                lm_offload = os.environ.get("ACESTEP_LM_OFFLOAD_TO_CPU", "false").lower() in ("1", "true", "yes")

                # Ensure the LM model is downloaded (not part of main model auto-download)
                try:
                    from acestep.model_downloader import ensure_lm_model
                    log.info("Ensuring LM model '%s' is available …", lm_model)
                    dl_ok, dl_msg = ensure_lm_model(
                        model_name=lm_model,
                        checkpoints_dir=CHECKPOINTS_DIR,
                    )
                    if dl_ok:
                        log.info("LM model check: %s", dl_msg)
                    else:
                        log.warning("LM model download failed: %s", dl_msg)
                except Exception as exc:
                    log.warning("LM model download check failed: %s", exc)

                log.info("Initializing LLM handler (model=%s, backend=%s) …", lm_model, lm_backend)
                try:
                    lm_status, lm_ok = llm.initialize(
                        checkpoint_dir=str(CHECKPOINTS_DIR),
                        lm_model_path=lm_model,
                        backend=lm_backend,
                        device=lm_device,
                        offload_to_cpu=lm_offload,
                    )
                    if lm_ok:
                        log.info("LLM handler ready: %s", lm_status)
                    else:
                        log.warning("LLM init returned not-ok: %s — proceeding without LLM", lm_status)
                except Exception as exc:
                    log.warning("LLM init failed: %s — proceeding without LLM", exc)
                _llm_handler = llm

            _init_done = True
            return True

        except Exception as exc:
            _init_error = str(exc)
            _init_done = True
            log.error("ACE-Step initialization failed: %s", exc)
            traceback.print_exc()
            return False


# ---------------------------------------------------------------------------
# Job store
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
        silence = b"\x00\x00\x00\x00" * num_frames
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
    JOBS.update(job_id, status="running", progress=0.01, message="Starting up …")

    output_dir = Path.home() / "Library" / "Application Support" / "Auralux" / "Generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_id}.wav"

    try:
        # If the model hasn't been initialized yet, run initialization in
        # a secondary thread so we can provide progress feedback.
        if not _init_done:
            JOBS.update(
                job_id,
                progress=0.02,
                message="Loading AI model (first run may take several minutes) …",
            )
            init_event = threading.Event()
            init_result: Dict[str, Any] = {}

            def _do_init() -> None:
                init_result["ok"] = _ensure_initialized()
                init_event.set()

            init_thread = threading.Thread(target=_do_init, daemon=True)
            init_thread.start()

            elapsed = 0.0
            tick = 2.0
            while not init_event.wait(timeout=tick):
                job = JOBS.get(job_id)
                if job and job.cancelled:
                    return
                elapsed += tick
                progress = min(0.08, 0.02 + elapsed * 0.0005)
                JOBS.update(
                    job_id,
                    progress=round(progress, 4),
                    message=f"Loading AI model … ({int(elapsed)}s elapsed)",
                )

            model_ready = init_result.get("ok", False)
        else:
            model_ready = _dit_handler is not None

        JOBS.update(job_id, progress=0.09, message="Model ready, preparing generation …")

        if model_ready and _dit_handler is not None:
            _run_real_inference(
                job_id=job_id,
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
    prompt: str,
    lyrics: str,
    tags: List[str],
    duration: float,
    variance: float,
    seed: Optional[int],
    output_path: Path,
) -> None:
    """Run ACE-Step v1.5 inference and write the output audio file.

    Because ``generate_music()`` is a blocking call with no built-in
    progress callback, we run it in a secondary thread and advance a
    synthetic progress bar on the current thread so the Swift client
    sees continuous updates while inference is running.
    """
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    caption = prompt
    if tags:
        caption = ", ".join(tags) + ". " + caption

    JOBS.update(job_id, progress=0.10, message="Preparing generation …")

    use_random = seed is None or seed < 0
    actual_seed = seed if not use_random else -1

    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics if lyrics else "",
        instrumental=not bool(lyrics),
        duration=float(duration) if duration > 0 else -1.0,
        seed=actual_seed,
        inference_steps=8,
        thinking=_llm_handler is not None and getattr(_llm_handler, "llm_initialized", False),
    )

    config = GenerationConfig(
        batch_size=1,
        use_random_seed=use_random,
        seeds=[actual_seed] if not use_random else None,
        audio_format="wav",
    )

    save_dir = str(output_path.parent)

    JOBS.update(job_id, progress=0.15, message="Running inference …")

    # Run the blocking inference in a secondary thread so we can
    # update synthetic progress on the current thread.
    inference_result: Dict[str, Any] = {}
    inference_done = threading.Event()

    def _do_inference() -> None:
        try:
            result = generate_music(
                dit_handler=_dit_handler,
                llm_handler=_llm_handler,
                params=params,
                config=config,
                save_dir=save_dir,
            )
            inference_result["value"] = result
        except Exception as exc:
            inference_result["error"] = exc
        finally:
            inference_done.set()

    worker = threading.Thread(target=_do_inference, daemon=True)
    worker.start()

    # Synthetic progress: smoothly advance from 0.15 → 0.90 while
    # inference runs.  Estimated total time scales with duration.
    estimated_seconds = max(30.0, duration * 2.5)
    progress_start = 0.15
    progress_end = 0.90
    elapsed = 0.0
    tick = 1.0

    while not inference_done.wait(timeout=tick):
        job = JOBS.get(job_id)
        if job and job.cancelled:
            log.info("Job %s cancelled during inference", job_id)
            return

        elapsed += tick
        fraction = min(elapsed / estimated_seconds, 1.0)
        current = progress_start + (progress_end - progress_start) * fraction
        pct = int(current * 100)
        JOBS.update(
            job_id,
            progress=round(current, 3),
            message=f"Generating audio … {pct}%",
        )

    if "error" in inference_result:
        raise inference_result["error"]

    result = inference_result["value"]

    if not result.success:
        raise RuntimeError(result.error or "Generation failed with no error message")

    JOBS.update(job_id, progress=0.95, message="Saving audio …")

    if result.audios and len(result.audios) > 0:
        audio_info = result.audios[0]
        generated_path = audio_info.get("path")

        if generated_path and Path(generated_path).exists():
            import shutil
            if Path(generated_path).resolve() != output_path.resolve():
                shutil.copy2(generated_path, output_path)
        elif "tensor" in audio_info:
            import torchaudio
            tensor = audio_info["tensor"]
            sr = audio_info.get("sample_rate", 48000)
            torchaudio.save(str(output_path), tensor, sr)
        else:
            raise RuntimeError("No audio data in generation result")
    else:
        raise RuntimeError("Generation returned empty audio list")


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


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "AuraluxServer/1.5"

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

    def do_GET(self) -> None:
        if self.path == "/health":
            dit_ready = _dit_handler is not None and _init_done
            llm_ready = _llm_handler is not None and getattr(_llm_handler, "llm_initialized", False)
            device = getattr(_dit_handler, "device", "unknown") if _dit_handler else "unknown"

            self._send_json({
                "status": "ok",
                "modelLoaded": dit_ready,
                "llmLoaded": llm_ready,
                "modelError": _init_error,
                "device": str(device),
                "engine": "ace-step-v1.5",
                "ditModel": os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo"),
                "llmModel": os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B"),
            })
            return

        if self.path.startswith("/jobs/"):
            job_id = self.path.split("/")[2]
            job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json({
                "jobID": job.id,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "audioPath": job.audio_path,
            })
            return

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
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

            self._send_json(
                {"jobID": job.id, "status": "queued", "message": "accepted"},
                HTTPStatus.ACCEPTED,
            )
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

        if self.path == "/models/download":
            threading.Thread(target=_ensure_initialized, daemon=True).start()
            self._send_json({"status": "download_started", "message": "Model download triggered"})
            return

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        log.debug(format, *args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Auralux API server (ACE-Step v1.5)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preload", action="store_true", help="Load model at startup instead of lazily")
    args = parser.parse_args()

    if args.preload:
        log.info("Preloading model …")
        _ensure_initialized()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log.info("Auralux API server listening on http://127.0.0.1:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down …")
        server.shutdown()


if __name__ == "__main__":
    main()
