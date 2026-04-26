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

MAX_REQUEST_BYTES = 1_000_000  # 1 MB cap on incoming request bodies

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auralux")
STARTED_AT = time.time()

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
# Monkey-patch: inference_mode → no_grad on MPS
# ---------------------------------------------------------------------------
# On MPS, torch.inference_mode() marks intermediate tensors as read-only.
# Several Metal compute shaders (mul_dense_scalar_float_float,
# masked_fill_scalar_strided_32bit, etc.) then crash with a fatal
# assertion because they try to bind those read-only buffers to
# write-enabled shader arguments.
#
# torch.no_grad() provides identical semantics for pure inference but does
# NOT mark tensors as read-only, neatly side-stepping the issue.  This
# global replacement is applied only when MPS is available.
try:
    import functools as _functools
    import torch as _torch_im

    if hasattr(_torch_im.backends, "mps") and _torch_im.backends.mps.is_available():
        _orig_inference_mode = _torch_im.inference_mode

        class _NoGradInferenceMode:
            """Drop-in torch.inference_mode replacement using no_grad."""

            def __init__(self, mode=True):
                self._inner = (
                    _torch_im.no_grad()
                    if mode
                    else _orig_inference_mode(mode=False)
                )

            def __enter__(self):
                return self._inner.__enter__()

            def __exit__(self, *args):
                return self._inner.__exit__(*args)

            def __call__(self, fn):
                @_functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    with self:
                        return fn(*args, **kwargs)
                return wrapper

        _torch_im.inference_mode = _NoGradInferenceMode
        log.info("Replaced torch.inference_mode with no_grad wrapper for MPS safety")
except ImportError:
    pass


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
# Monkey-patch: text-encoder embeddings on MPS
# ---------------------------------------------------------------------------
# The same ``mul_dense_scalar_float_float`` Metal shader bug that affects
# audio-code decoding also triggers during the text-encoder forward pass
# inside ``infer_text_embeddings`` and ``infer_lyric_embeddings``
# (ConditioningEmbedMixin).  The call runs under torch.inference_mode()
# which marks tensors read-only, and the Metal shader then tries to bind
# a read-only buffer to a write-enabled argument — causing a fatal process
# crash.
#
# The fix temporarily moves the text encoder to CPU for the forward pass,
# then puts it back on MPS.  The text encoder is small (0.6B) so the
# performance impact is minimal.

def _patch_mps_text_encoder_embed() -> None:
    """Replace infer_text_embeddings / infer_lyric_embeddings with CPU-safe variants."""
    try:
        import torch
        from acestep.core.generation.handler.conditioning_embed import ConditioningEmbedMixin
    except ImportError:
        return

    _orig_infer_text = ConditioningEmbedMixin.infer_text_embeddings
    _orig_infer_lyric = ConditioningEmbedMixin.infer_lyric_embeddings

    def _cpu_safe_infer_text(self, text_token_idss):
        if not hasattr(self, "device") or str(self.device).split(":")[0] != "mps":
            return _orig_infer_text(self, text_token_idss)

        mps_dev = self.device
        self.text_encoder.cpu()
        try:
            cpu_ids = text_token_idss.cpu() if hasattr(text_token_idss, "cpu") else text_token_idss
            with torch.inference_mode():
                result = self.text_encoder(input_ids=cpu_ids, lyric_attention_mask=None).last_hidden_state
            return result.to(mps_dev)
        finally:
            self.text_encoder.to(mps_dev)

    def _cpu_safe_infer_lyric(self, lyric_token_ids):
        if not hasattr(self, "device") or str(self.device).split(":")[0] != "mps":
            return _orig_infer_lyric(self, lyric_token_ids)

        mps_dev = self.device
        self.text_encoder.cpu()
        try:
            cpu_ids = lyric_token_ids.cpu() if hasattr(lyric_token_ids, "cpu") else lyric_token_ids
            with torch.inference_mode():
                result = self.text_encoder.embed_tokens(cpu_ids)
            return result.to(mps_dev)
        finally:
            self.text_encoder.to(mps_dev)

    ConditioningEmbedMixin.infer_text_embeddings = _cpu_safe_infer_text
    ConditioningEmbedMixin.infer_lyric_embeddings = _cpu_safe_infer_lyric
    log.info("Patched infer_text_embeddings/infer_lyric_embeddings for MPS safety (CPU fallback)")


# ---------------------------------------------------------------------------
# Monkey-patch: prepare_condition (DiT condition encoder) on MPS
# ---------------------------------------------------------------------------
# The DiT condition encoder (AceStepConditionEncoder) contains transformer
# layers whose attention-scaling scalar multiplications trigger the same
# ``mul_dense_scalar_float_float`` Metal shader fatal assertion.  The bug
# originates from MPS allocating model-parameter Metal buffers as read-only.
#
# Because this is a device-level buffer property (not caused by
# inference_mode), the only reliable fix is to run the encoder on CPU.
# We move only the sub-modules used by prepare_condition (encoder,
# tokenizer, detokenizer) — NOT the large DiT decoder — so the overhead
# is small, especially on Apple Silicon unified memory.
#
# IMPORTANT: The model is loaded via AutoModel.from_pretrained with
# trust_remote_code=True, which dynamically creates the class from the
# checkpoint directory.  This means the class object is DIFFERENT from the
# one importable via ``acestep.models.turbo.modeling_acestep_v15_turbo``.
# We must therefore patch the actual class of the loaded model instance,
# not the package-level import.

def _patch_mps_prepare_condition(model_instance) -> None:
    """Run prepare_condition on CPU to avoid MPS Metal shader bugs.

    Must be called AFTER the model is loaded so we patch the correct
    (dynamically-created) class, not the package-level import.
    """
    try:
        import torch
    except ImportError:
        return

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    model_cls = model_instance.__class__
    _orig = model_cls.prepare_condition

    def _cpu_safe_prepare_condition(self, *args, **kwargs):
        mps_device = None
        for a in args:
            if isinstance(a, torch.Tensor) and a.device.type == "mps":
                mps_device = a.device
                break
        if mps_device is None:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor) and v.device.type == "mps":
                    mps_device = v.device
                    break
        if mps_device is None:
            return _orig(self, *args, **kwargs)

        modules = [self.encoder]
        for attr in ("tokenizer", "detokenizer"):
            mod = getattr(self, attr, None)
            if mod is not None:
                modules.append(mod)

        for m in modules:
            m.cpu()

        cpu_args = tuple(
            a.cpu() if isinstance(a, torch.Tensor) else a for a in args
        )
        cpu_kwargs = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        try:
            result = _orig(self, *cpu_args, **cpu_kwargs)
            if isinstance(result, tuple):
                return tuple(
                    r.to(mps_device) if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            return (
                result.to(mps_device)
                if isinstance(result, torch.Tensor)
                else result
            )
        finally:
            for m in modules:
                m.to(mps_device)

    model_cls.prepare_condition = _cpu_safe_prepare_condition
    log.info(
        "Patched %s.prepare_condition for MPS safety (CPU fallback)",
        model_cls.__name__,
    )


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
            _patch_mps_text_encoder_embed()

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

            # Patch prepare_condition on the ACTUAL model instance.
            # The model class is dynamically created by AutoModel.from_pretrained
            # with trust_remote_code=True, so we must patch after loading.
            if dit.model is not None:
                _patch_mps_prepare_condition(dit.model)

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

    def status_counts(self) -> Dict[str, int]:
        with self._lock:
            counts: Dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            return counts


JOBS = JobStore()


def _process_stats() -> Dict[str, Any]:
    pid = os.getpid()
    stats: Dict[str, Any] = {
        "pid": pid,
        "uptimeSeconds": round(time.time() - STARTED_AT, 1),
        "activeThreads": threading.active_count(),
        "jobCounts": JOBS.status_counts(),
    }

    try:
        import psutil
        proc = psutil.Process(pid)
        stats["cpuPercent"] = proc.cpu_percent(interval=None)
        stats["memoryRSSMB"] = round(proc.memory_info().rss / (1024 * 1024), 1)
    except Exception as exc:
        stats["statsError"] = str(exc)

    return stats


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

        job = JOBS.get(job_id)
        if job and job.cancelled:
            JOBS.update(
                job_id,
                status="cancelled",
                progress=0.0,
                message="Job cancelled",
            )
            return

        JOBS.update(
            job_id,
            status="completed",
            progress=1.0,
            message=f"Generated track for: {prompt[:64]}",
            audio_path=str(output_path),
        )
    except Exception as exc:
        log.error("Generation failed for job %s: %s\n%s", job_id, exc, traceback.format_exc())
        JOBS.update(
            job_id,
            status="failed",
            message=f"Generation failed: {type(exc).__name__}: {exc}",
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
            log.info("Job %s cancelled during inference — waiting for inference thread", job_id)
            inference_done.wait()  # wait for the thread to finish before returning
            JOBS.update(job_id, status="cancelled", progress=0.0, message="Job cancelled")
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

    # Check cancellation one final time — may have been set while inference was finishing.
    job = JOBS.get(job_id)
    if job and job.cancelled:
        JOBS.update(job_id, status="cancelled", progress=0.0, message="Job cancelled")
        return

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

    def _read_json(self) -> Optional[Dict[str, object]]:
        """Read and parse the request body as JSON.

        Returns the parsed dict, or None when a response has already been sent
        to the client (caller must return immediately without sending another).
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return {}
        if length > MAX_REQUEST_BYTES:
            self._send_json(
                {"error": f"request body too large (max {MAX_REQUEST_BYTES} bytes)"},
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json({"error": f"invalid JSON: {exc}"}, HTTPStatus.BAD_REQUEST)
            return None

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
                "stats": _process_stats(),
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
            if payload is None:
                return

            prompt = str(payload.get("prompt", "")).strip()
            lyrics = str(payload.get("lyrics", ""))
            tags = list(payload.get("tags", []))

            try:
                duration = float(payload.get("duration", 30))
                if not (0 < duration <= 600):
                    raise ValueError(f"duration must be between 0 and 600, got {duration}")
            except (TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

            try:
                variance = float(payload.get("variance", 0.5))
                if not (0 <= variance <= 2):
                    raise ValueError(f"variance must be between 0 and 2, got {variance}")
            except (TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

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
        msg = format % args
        # Log non-2xx responses at INFO so HTTP errors are visible without
        # changing the global log level.
        if args and isinstance(args[1], str) and not args[1].startswith("2"):
            log.info(msg)
        else:
            log.debug(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class AuraluxHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with socket reuse and daemon request threads.

    allow_reuse_address prevents TIME_WAIT from blocking a restart after a crash.
    daemon_threads ensures request threads are torn down when the main thread exits.
    """
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Auralux API server (ACE-Step v1.5)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preload", action="store_true", help="Load model at startup instead of lazily")
    args = parser.parse_args()

    if args.preload:
        log.info("Preloading model …")
        _ensure_initialized()

    server = AuraluxHTTPServer(("127.0.0.1", args.port), Handler)
    log.info("Auralux API server listening on http://127.0.0.1:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down …")
        server.shutdown()


if __name__ == "__main__":
    main()
