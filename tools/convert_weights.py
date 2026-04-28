#!/usr/bin/env python3
"""Convert ACE-Step v1.5 checkpoints to MLX-native safetensors format.

Downloads weights directly from HuggingFace, converts key names and tensor
layouts for the Swift MLX runtime, and writes to the app's model directory.

Usage:
    python tools/convert_weights.py [--output-dir DIR] [--dtype float16|float32]
    python tools/convert_weights.py --inspect-only

Default output: ~/Library/Application Support/Auralux/Models/ace-step-v1.5-mlx/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

def _check_deps() -> bool:
    missing = []
    for mod in ("numpy", "safetensors", "huggingface_hub"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}", file=sys.stderr)
        print(f"Run: pip install {' '.join(missing)}", file=sys.stderr)
        return False
    return True

# ---------------------------------------------------------------------------
# Key-name mapping: PyTorch DiT → Swift module hierarchy
# ---------------------------------------------------------------------------
# Key remapping: ACE-Step v1.5 Turbo checkpoint → Swift module hierarchy
#
# Checkpoint top-level prefixes:
#   decoder.*                → decoder.*
#   encoder.lyric_encoder.*  → lyricEncoder.*
#   detokenizer.*            → detokenizer.*
#   null_condition_emb       → nullConditionEmb
#   tokenizer.*              → (skipped)
# ---------------------------------------------------------------------------

def _snake_to_camel(s: str) -> str:
    """Convert a single snake_case identifier to camelCase."""
    parts = s.split("_")
    if len(parts) == 1:
        return s
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _camel_path(path: str) -> str:
    """Convert every dot-separated segment of a weight path to camelCase."""
    return ".".join(_snake_to_camel(seg) for seg in path.split("."))


def _remap_attn_key(rest: str) -> str:
    """q_proj → qProj, k_norm → kNorm, o_proj → oProj, etc."""
    return _camel_path(rest)


def _remap_encoder_layer_key(rest: str) -> str:
    """Remap keys inside an AceStepEncoderLayer."""
    seg_map = {
        "input_layernorm":          "inputLayernorm",
        "post_attention_layernorm": "postAttentionLayernorm",
        "self_attn":                "selfAttn",
        "mlp":                      "mlp",
    }
    # first segment is the sub-module name
    parts = rest.split(".", maxsplit=1)
    head  = parts[0]
    tail  = ("." + _camel_path(parts[1])) if len(parts) > 1 else ""
    return seg_map.get(head, _snake_to_camel(head)) + tail


def _remap_decoder_layer_key(rest: str) -> str:
    """Remap keys inside an AceStepDiTLayer."""
    parts = rest.split(".", maxsplit=1)
    head  = parts[0]
    tail  = ("." + _camel_path(parts[1])) if len(parts) > 1 else ""
    seg_map = {
        "self_attn_norm":  "selfAttnNorm",
        "self_attn":       "selfAttn",
        "cross_attn_norm": "crossAttnNorm",
        "cross_attn":      "crossAttn",
        "mlp_norm":        "mlpNorm",
        "mlp":             "mlp",
        "scale_shift_table": "scaleShiftTable",
    }
    return seg_map.get(head, _snake_to_camel(head)) + tail


def _remap_decoder_key(rest: str) -> str:
    """Remap keys under the 'decoder' prefix."""
    # layers.N.*
    m = re.match(r"^layers\.(\d+)\.(.+)$", rest)
    if m:
        idx, sub = m.group(1), m.group(2)
        return f"layers.{idx}.{_remap_decoder_layer_key(sub)}"

    # proj_in.1.*  → projIn.*   (strip Sequential index .1.)
    if rest.startswith("proj_in.1."):
        return "projIn." + rest[len("proj_in.1."):]
    # proj_out.1.* → projOut.*
    if rest.startswith("proj_out.1."):
        return "projOut." + rest[len("proj_out.1."):]

    # Bare scale_shift_table
    if rest == "scale_shift_table":
        return "scaleShiftTable"

    # Named sub-modules: snake_case first segment → camelCase
    top_map = {
        "time_embed_r": "timeEmbedR",
        "time_embed":   "timeEmbed",
        "condition_embedder": "conditionEmbedder",
        "norm_out":     "normOut",
    }
    parts = rest.split(".", maxsplit=1)
    head  = parts[0]
    tail  = ("." + _camel_path(parts[1])) if len(parts) > 1 else ""
    return top_map.get(head, _snake_to_camel(head)) + tail


def _remap_dit_key(key: str) -> Optional[str]:
    """Map a checkpoint key to its Swift MLX module path. Returns None to skip."""

    # ── Audio tokenizer (tokenizer.*) → audioTokenizer.* ───────────────────
    # tokenizer.audio_acoustic_proj.*           → audioTokenizer.audioAcousticProj.*
    # tokenizer.attention_pooler.embed_tokens.* → audioTokenizer.attentionPooler.embedTokens.*
    # tokenizer.attention_pooler.special_token  → audioTokenizer.attentionPooler.specialToken
    # tokenizer.attention_pooler.norm.*         → audioTokenizer.attentionPooler.norm.*
    # tokenizer.attention_pooler.layers.N.*     → audioTokenizer.attentionPooler.layers.N.*
    # tokenizer.quantizer.project_in.*          → audioTokenizer.quantizer.projectIn.*
    # tokenizer.quantizer.project_out.*         → audioTokenizer.quantizer.projectOut.*
    if key.startswith("tokenizer."):
        rest = key[len("tokenizer."):]
        if rest.startswith("audio_acoustic_proj."):
            return "audioTokenizer.audioAcousticProj." + rest[len("audio_acoustic_proj."):]
        if rest.startswith("attention_pooler."):
            sub = rest[len("attention_pooler."):]
            m = re.match(r"^layers\.(\d+)\.(.+)$", sub)
            if m:
                idx, layer_sub = m.group(1), m.group(2)
                return f"audioTokenizer.attentionPooler.layers.{idx}.{_remap_encoder_layer_key(layer_sub)}"
            top_map = {
                "embed_tokens":  "embedTokens",
                "special_token": "specialToken",
                "norm":          "norm",
            }
            parts = sub.split(".", maxsplit=1)
            head  = parts[0]
            tail  = ("." + _camel_path(parts[1])) if len(parts) > 1 else ""
            return "audioTokenizer.attentionPooler." + top_map.get(head, _snake_to_camel(head)) + tail
        if rest.startswith("quantizer."):
            sub = rest[len("quantizer."):]
            sub = sub.replace("project_in.", "projectIn.").replace("project_out.", "projectOut.")
            return "audioTokenizer.quantizer." + sub
        # rotary_emb buffers etc. — runtime-derived in Swift, not loaded.
        return None

    # ── Skip components not needed for basic inference ─────────────────────
    if key.startswith("encoder.attention_pooler."):  # only the audio tokenizer pooler is plumbed
        return None

    # ── Timbre encoder (encoder.timbre_encoder.*) → timbreEncoder.* ────────
    if key.startswith("encoder.timbre_encoder."):
        rest = key[len("encoder.timbre_encoder."):]
        m = re.match(r"^layers\.(\d+)\.(.+)$", rest)
        if m:
            idx, sub = m.group(1), m.group(2)
            return f"timbreEncoder.layers.{idx}.{_remap_encoder_layer_key(sub)}"
        top_map = {
            "embed_tokens": "embedTokens",
            "special_token": "specialToken",
            "norm": "norm",
        }
        parts = rest.split(".", maxsplit=1)
        head  = parts[0]
        tail  = ("." + _camel_path(parts[1])) if len(parts) > 1 else ""
        return "timbreEncoder." + top_map.get(head, _snake_to_camel(head)) + tail

    # ── Null condition embedding ────────────────────────────────────────────
    if key == "null_condition_emb":
        return "nullConditionEmb"

    # ── Text projector (encoder.text_projector.weight, no bias) ─────────────
    # Linear(text_hidden_dim=1024 → hidden_size=2048).
    if key.startswith("encoder.text_projector."):
        return "textProjector." + key[len("encoder.text_projector."):]

    # ── Decoder (AceStepDiTModel) ───────────────────────────────────────────
    if key.startswith("decoder."):
        rest = key[len("decoder."):]
        return "decoder." + _remap_decoder_key(rest)

    # ── Lyric encoder (encoder.lyric_encoder.*) → lyricEncoder.* ───────────
    if key.startswith("encoder.lyric_encoder."):
        rest = key[len("encoder.lyric_encoder."):]
        parts = rest.split(".", maxsplit=1)
        head  = parts[0]
        tail  = ("." + parts[1]) if len(parts) > 1 else ""

        # layers.N.* → recurse through encoder-layer remapper
        m = re.match(r"^layers\.(\d+)\.(.+)$", rest)
        if m:
            idx, sub = m.group(1), m.group(2)
            return f"lyricEncoder.layers.{idx}.{_remap_encoder_layer_key(sub)}"

        top_map = {"embed_tokens": "embedTokens", "norm": "norm"}
        return "lyricEncoder." + top_map.get(head, _snake_to_camel(head)) + (
            "." + _camel_path(tail[1:]) if tail else ""
        )

    # ── Detokenizer (AudioTokenDetokenizer) ────────────────────────────────
    if key.startswith("detokenizer."):
        rest = key[len("detokenizer."):]

        m = re.match(r"^layers\.(\d+)\.(.+)$", rest)
        if m:
            idx, sub = m.group(1), m.group(2)
            return f"detokenizer.layers.{idx}.{_remap_encoder_layer_key(sub)}"

        top_map = {
            "embed_tokens": "embedTokens",
            "special_tokens": "specialTokens",
            "proj_out": "projOut",
            "norm": "norm",
        }
        parts = rest.split(".", maxsplit=1)
        head  = parts[0]
        tail  = ("." + parts[1]) if len(parts) > 1 else ""
        return "detokenizer." + top_map.get(head, _snake_to_camel(head)) + (
            "." + _camel_path(tail[1:]) if tail else ""
        )

    print(f"  [unknown dit] {key}", file=sys.stderr)
    return None


# Block-level remap shared by the 5Hz LM and the Qwen3-Embedding text encoder.
# Both are Qwen3-derived: same RMSNorm + GQA + RoPE + SwiGLU MLP layout.
_QWEN3_BLOCK_MAP: Dict[str, str] = {
    "input_layernorm.weight":          "inputLayernorm.weight",
    "post_attention_layernorm.weight": "postAttentionLayernorm.weight",
    "self_attn.q_proj.weight":         "selfAttn.qProj.weight",
    "self_attn.q_proj.bias":           "selfAttn.qProj.bias",
    "self_attn.k_proj.weight":         "selfAttn.kProj.weight",
    "self_attn.k_proj.bias":           "selfAttn.kProj.bias",
    "self_attn.v_proj.weight":         "selfAttn.vProj.weight",
    "self_attn.v_proj.bias":           "selfAttn.vProj.bias",
    "self_attn.o_proj.weight":         "selfAttn.oProj.weight",
    "self_attn.q_norm.weight":         "selfAttn.qNorm.weight",
    "self_attn.k_norm.weight":         "selfAttn.kNorm.weight",
    "mlp.gate_proj.weight":            "mlp.gateProj.weight",
    "mlp.up_proj.weight":              "mlp.upProj.weight",
    "mlp.down_proj.weight":            "mlp.downProj.weight",
}


def _remap_lm_key(key: str) -> Optional[str]:
    """Map a Qwen2/Qwen3-style LM key (with `model.` prefix) to Swift module path."""
    lm_map = {
        "model.norm.weight": "norm.weight",
        "lm_head.weight":    "lmHead.weight",
    }
    if key in lm_map:
        return lm_map[key]

    m = re.match(r"^model\.embed_tokens\.(.+)$", key)
    if m:
        return "embedTokens." + m.group(1)

    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", key)
    if m:
        idx, rest = m.group(1), m.group(2)
        if rest in _QWEN3_BLOCK_MAP:
            return f"layers.{idx}.{_QWEN3_BLOCK_MAP[rest]}"
        print(f"  [skip lm] layers.{idx}.{rest}", file=sys.stderr)
        return None

    print(f"  [unknown lm] {key}", file=sys.stderr)
    return None


def _remap_text_encoder_key(key: str) -> Optional[str]:
    """Map a Qwen3-Embedding-0.6B key to Swift module path.

    Differences vs the 5Hz LM dump:
      * No `model.` prefix on top-level keys.
      * Tied word embeddings, so no `lm_head.weight`.
    """
    if key == "norm.weight":
        return "norm.weight"

    if key == "lm_head.weight":
        # Tied to embed_tokens; safetensors should not include this, but tolerate.
        return None

    m = re.match(r"^embed_tokens\.(.+)$", key)
    if m:
        return "embedTokens." + m.group(1)

    m = re.match(r"^layers\.(\d+)\.(.+)$", key)
    if m:
        idx, rest = m.group(1), m.group(2)
        if rest in _QWEN3_BLOCK_MAP:
            return f"layers.{idx}.{_QWEN3_BLOCK_MAP[rest]}"
        print(f"  [skip text] layers.{idx}.{rest}", file=sys.stderr)
        return None

    print(f"  [unknown text] {key}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# VAE key mapping: PyTorch Oobleck VAE → Swift DCHiFiGAN{Decoder,Encoder}
# ---------------------------------------------------------------------------
#
# Decoder prefix "decoder.*" → Swift module tree under `decoder.*`:
#   decoder.conv1.*           → decoder.conv1.*
#   decoder.block.N.conv_t1.* → decoder.blocks.N.convT1.*
#   decoder.block.N.res_unitM.*→ decoder.blocks.N.resUnitM.*
#   decoder.block.N.snake1.*  → decoder.blocks.N.snake1.*
#   decoder.snake1.*          → decoder.snake1.*
#   decoder.conv2.*           → decoder.conv2.*
#
# Encoder prefix "encoder.*" → Swift module tree under `encoder.*`:
#   encoder.conv1.*           → encoder.conv1.*
#   encoder.block.N.conv1.*   → encoder.blocks.N.conv1.*
#   encoder.block.N.res_unitM → encoder.blocks.N.resUnitM.*
#   encoder.block.N.snake1.*  → encoder.blocks.N.snake1.*
#   encoder.snake1.*          → encoder.snake1.*
#   encoder.conv2.*           → encoder.conv2.*
# ---------------------------------------------------------------------------

def _remap_vae_key(key: str) -> Optional[str]:
    """Map a VAE encoder/decoder key (after weight_norm fusion) to its Swift path."""
    for prefix, swift_root in (("decoder.", "decoder"), ("encoder.", "encoder")):
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]

        # Top-level tensors: conv1, conv2, snake1 sit directly under the root.
        for tok in ("conv1.", "conv2.", "snake1."):
            if rest.startswith(tok):
                return f"{swift_root}.{rest}"

        # block.N.* → blocks.N.* (with sub-name remapping for convT1/resUnitM).
        m = re.match(r"^block\.(\d+)\.(.+)$", rest)
        if m:
            idx, sub = m.group(1), m.group(2)
            sub = sub.replace("conv_t1.", "convT1.")
            sub = re.sub(r"res_unit(\d+)\.", lambda x: f"resUnit{x.group(1)}.", sub)
            return f"{swift_root}.blocks.{idx}.{sub}"

        print(f"  [unknown vae] {key}", file=sys.stderr)
        return None
    return None  # not encoder/decoder — silently skip


# ---------------------------------------------------------------------------
# Conv weight transposition
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HuggingFace download helpers
# ---------------------------------------------------------------------------

def _download(repo_id: str, filename: str, local_dir: Path, token: Optional[str]) -> Path:
    from huggingface_hub import hf_hub_download
    print(f"  Downloading {repo_id}/{filename} …")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        token=token,
    )
    return Path(path)


def _download_dir(repo_id: str, subdir: Optional[str], local_dir: Path, token: Optional[str],
                  patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download
    allow = patterns
    ignore = None
    print(f"  Downloading {repo_id} {'/' + subdir if subdir else ''} …")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow,
        ignore_patterns=ignore,
        token=token,
    )


# ---------------------------------------------------------------------------
# bfloat16-aware safetensors loader (numpy has no bfloat16)
# ---------------------------------------------------------------------------

def _load_safetensors_numpy(path: Path) -> Dict[str, Any]:
    """Read a safetensors file into a dict of numpy float32 arrays.

    Handles BF16, F16, F32, and I64 dtypes without requiring torch.
    BF16 tensors are widened to F32 via uint16 bit-shift (lossless for
    the exponent/mantissa bits that BF16 carries).
    """
    import numpy as np

    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header: Dict[str, Any] = json.loads(f.read(header_len))
        data_start = 8 + header_len
        raw_all = f.read()  # rest of file = tensor data

    out: Dict[str, Any] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dtype_str = meta["dtype"]
        shape = tuple(meta["shape"])
        lo, hi = meta["data_offsets"]
        raw = raw_all[lo:hi]

        if dtype_str == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        elif dtype_str == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        elif dtype_str == "F64":
            arr = np.frombuffer(raw, dtype=np.float64).reshape(shape).astype(np.float32)
        elif dtype_str in ("I64", "I32", "I16", "I8", "U8", "BOOL"):
            # Keep integer tensors as-is (embeddings, position ids, etc.)
            np_dtype = {"I64": np.int64, "I32": np.int32, "I16": np.int16,
                        "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}[dtype_str]
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
        else:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_str} (key={key})")

        out[key] = arr
    return out


# ---------------------------------------------------------------------------
# Sharded safetensors loader (XL checkpoints: 4 × ~5 GB shards)
# ---------------------------------------------------------------------------

def _load_sharded_safetensors_numpy(shard_dir: Path) -> Dict[str, Any]:
    """Load a sharded safetensors checkpoint (model.safetensors.index.json + shards).

    Reads the index JSON, discovers all unique shard filenames, loads each shard
    sequentially, and merges into a single dict (same contract as
    `_load_safetensors_numpy`).
    """
    index_path = shard_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    # weight_map: {param_name: shard_filename}
    weight_map: Dict[str, str] = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))
    print(f"  Found {len(shard_files)} shard(s) covering {len(weight_map)} tensors")

    merged: Dict[str, Any] = {}
    for shard_name in shard_files:
        shard_path = shard_dir / shard_name
        print(f"  Loading shard {shard_name} …")
        merged.update(_load_safetensors_numpy(shard_path))

    return merged


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _needs_conv1d_transpose(new_key: str, shape: tuple) -> bool:
    """True for Conv1d weight: PyTorch [out, in, k] → MLX [out, k, in]."""
    return len(shape) == 3 and new_key.endswith(".weight") and (
        "projIn.weight" in new_key or "projOut.weight" in new_key
    )

def _needs_convT1d_transpose(new_key: str, shape: tuple) -> bool:
    """True for ConvTranspose1d weight: PyTorch [in, out, k] → MLX [out, k, in]."""
    return len(shape) == 3 and new_key.endswith(".weight") and "projOut.weight" in new_key


def convert_dit(src: Path, output_dir: Path, dtype_str: str, sharded: bool = False) -> bool:
    """Convert a DiT checkpoint (single file or sharded) to MLX safetensors.

    `src` is either:
      - a .safetensors file path  (sharded=False)
      - the directory containing model.safetensors.index.json  (sharded=True)
    """
    import numpy as np
    from safetensors.numpy import save_file as save_np

    dtype_map = {"float32": np.float32, "float16": np.float16}
    target_dtype = dtype_map.get(dtype_str, np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    converted: Dict[str, Any] = {}
    skipped = 0

    if sharded:
        print(f"  Loading sharded checkpoint from {src} …")
        source = _load_sharded_safetensors_numpy(src)
    else:
        print(f"  Converting {src.name} …")
        source = _load_safetensors_numpy(src)
    keys = list(source.keys())
    print(f"  Source tensors: {len(keys)}")
    for key in keys:
        tensor = source[key]
        shape  = tuple(tensor.shape)
        new_key = _remap_dit_key(key)
        if new_key is None:
            skipped += 1
            continue

        # Conv1d (proj_in): PyTorch [out, in, k] → MLX [out, k, in]
        if key == "decoder.proj_in.1.weight" and len(shape) == 3:
            tensor = tensor.transpose(0, 2, 1)
        # ConvTransposed1d (proj_out): PyTorch [in, out, k] → MLX [out, k, in]
        elif key == "decoder.proj_out.1.weight" and len(shape) == 3:
            tensor = tensor.transpose(1, 2, 0)

        if tensor.dtype in (np.float32, np.float16, np.float64):
            tensor = tensor.astype(target_dtype)
        converted[new_key] = np.ascontiguousarray(tensor)

    out_path = output_dir / "dit_weights.safetensors"
    print(f"  Saving {len(converted)} tensors → {out_path} ({skipped} skipped) …")
    save_np(converted, str(out_path))

    saved_keys = list(_load_safetensors_numpy(out_path).keys())
    print(f"  ✓ DiT: {len(saved_keys)} tensors written ({skipped} skipped)")
    return True


def convert_silence_latent(src_pt: Path, output_dir: Path, dtype_str: str) -> bool:
    """Convert upstream silence_latent.pt to MLX safetensors.

    ACE-Step stores this PyTorch tensor as shape [1, 64, T].  The Swift engine
    consumes [1, T, 64] so it can concatenate it with a [1, T, 64] chunk mask.
    """
    import numpy as np
    from safetensors.numpy import save_file as save_np

    np_dtype = np.float16 if dtype_str == "float16" else np.float32

    print(f"  Converting {src_pt.name} …")
    with zipfile.ZipFile(src_pt, "r") as zf:
        data_member = next((name for name in zf.namelist() if name.endswith("/data/0")), None)
        if data_member is None:
            raise ValueError(f"{src_pt} does not look like a torch zip tensor archive")
        raw = zf.read(data_member)

    tensor = np.frombuffer(raw, dtype="<f4").copy()
    if tensor.size % 64 != 0:
        raise ValueError(f"silence_latent element count {tensor.size} is not divisible by 64")

    frames = tensor.size // 64
    tensor = tensor.reshape(1, 64, frames).transpose(0, 2, 1)
    tensor = np.ascontiguousarray(tensor.astype(np_dtype))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "silence_latent.safetensors"
    print(f"  Saving silence_latent {list(tensor.shape)} → {out_path} …")
    save_np({"silence_latent": tensor}, str(out_path))
    print("  ✓ Silence latent written")
    return True


def convert_text_encoder(src_safetensors: Path, output_dir: Path, dtype_str: str) -> bool:
    """Convert Qwen3-Embedding-0.6B to MLX-native safetensors.

    Used by the DiT for text/lyric conditioning (`encoder.lyric_encoder.embed_tokens` and
    `encoder.text_projector` consume 1024-dim hidden states from this model).
    """
    import numpy as np
    from safetensors.numpy import save_file as save_np

    dtype_map = {"float32": np.float32, "float16": np.float16}
    target_dtype = dtype_map.get(dtype_str, np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    converted: Dict[str, Any] = {}
    skipped = 0

    print(f"  Converting {src_safetensors.name} …")
    source = _load_safetensors_numpy(src_safetensors)
    keys = list(source.keys())
    print(f"  Source tensors: {len(keys)}")
    for key in keys:
        tensor = source[key]
        new_key = _remap_text_encoder_key(key)
        if new_key is None:
            skipped += 1
            continue
        if tensor.dtype in (np.float32, np.float16, np.float64):
            tensor = tensor.astype(target_dtype)
        converted[new_key] = np.ascontiguousarray(tensor)

    out_path = output_dir / "text_weights.safetensors"
    print(f"  Saving {len(converted)} tensors → {out_path} ({skipped} skipped) …")
    save_np(converted, str(out_path))

    saved_keys = list(_load_safetensors_numpy(out_path).keys())
    print(f"  ✓ Text encoder: {len(saved_keys)} tensors written ({skipped} skipped)")
    return True


def copy_text_tokenizer(src_dir: Path, output_dir: Path) -> None:
    """Copy Qwen3-Embedding tokenizer files with `text_` prefix to avoid LM-tokenizer collision."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "special_tokens_map.json", "added_tokens.json",
                 "config.json"]:
        src = src_dir / name
        if src.exists():
            dst = output_dir / ("text_" + name)
            shutil.copy2(src, dst)
            print(f"  Copied → {dst.name}")


def convert_lm(src_safetensors: Path, output_dir: Path, dtype_str: str) -> bool:
    import numpy as np
    from safetensors.numpy import save_file as save_np
    from safetensors import safe_open

    dtype_map = {"float32": np.float32, "float16": np.float16}
    target_dtype = dtype_map.get(dtype_str, np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    converted: Dict[str, Any] = {}
    skipped = 0

    print(f"  Converting {src_safetensors.name} …")
    source = _load_safetensors_numpy(src_safetensors)
    keys = list(source.keys())
    print(f"  Source tensors: {len(keys)}")
    for key in keys:
        tensor = source[key]
        new_key = _remap_lm_key(key)
        if new_key is None:
            skipped += 1
            continue
        if tensor.dtype in (np.float32, np.float16, np.float64):
            tensor = tensor.astype(target_dtype)
        converted[new_key] = np.ascontiguousarray(tensor)

    out_path = output_dir / "lm_weights.safetensors"
    print(f"  Saving {len(converted)} tensors → {out_path} ({skipped} skipped) …")
    save_np(converted, str(out_path))

    saved_keys = list(_load_safetensors_numpy(out_path).keys())
    print(f"  ✓ LM: {len(saved_keys)} tensors written ({skipped} skipped)")
    return True


def _fuse_weight_norm(v: "np.ndarray", g: "np.ndarray") -> "np.ndarray":
    """Fuse PyTorch weight_norm: w = g * v / ‖v‖₂  (norm over all dims except axis 0)."""
    import numpy as np
    norm = np.sqrt((v * v).sum(axis=tuple(range(1, v.ndim)), keepdims=True))
    return g * v / (norm + 1e-8)


def convert_vae(src_safetensors: Path, output_dir: Path, dtype_str: str) -> bool:
    """Convert Oobleck VAE encoder + decoder weights to MLX format.

    - Both encoder.* and decoder.* keys are kept. Encoder is needed for the
      cover/repaint/extract modes that consume user-provided audio.
    - PyTorch weight_norm (weight_v + weight_g pairs) is fused into a single weight.
    - Conv1d weights are transposed [out,in,k] → [out,k,in] for MLX.
    - ConvTranspose1d weights are transposed [in,out,k] → [out,k,in] for MLX.
    - Snake alpha/beta parameters are squeezed [1,C,1] → [C] for MLX's NLC layout.
    """
    import numpy as np
    from safetensors.numpy import save_file as save_np

    np_dtype = np.float16 if dtype_str == "float16" else np.float32

    print(f"  Reading {src_safetensors.name} …")
    tensors = _load_safetensors_numpy(src_safetensors)

    out: Dict[str, Any] = {}
    consumed: set = set()

    for key in sorted(tensors.keys()):
        if key in consumed:
            continue
        if not (key.startswith("decoder.") or key.startswith("encoder.")):
            continue  # skip anything outside encoder/decoder

        if key.endswith(".weight_v"):
            base  = key[: -len(".weight_v")]
            g_key = base + ".weight_g"
            v = tensors[key]
            g = tensors.get(g_key)
            consumed.update({key, g_key})

            w = _fuse_weight_norm(v, g) if g is not None else v

            # ConvTranspose1d: PyTorch [in,out,k] → MLX [out,k,in]
            # Conv1d:          PyTorch [out,in,k] → MLX [out,k,in]
            is_transpose = ".conv_t1" in base
            w = w.transpose(1, 2, 0) if is_transpose else w.transpose(0, 2, 1)

            mlx_key = _remap_vae_key(base + ".weight")
            if mlx_key:
                out[mlx_key] = np.ascontiguousarray(w.astype(np_dtype))

        elif key.endswith(".weight_g"):
            consumed.add(key)  # handled alongside .weight_v

        elif key.endswith(".bias"):
            consumed.add(key)
            mlx_key = _remap_vae_key(key)
            if mlx_key:
                out[mlx_key] = np.ascontiguousarray(tensors[key].astype(np_dtype))

        elif key.endswith(".alpha") or key.endswith(".beta"):
            consumed.add(key)
            # [1, C, 1] → [C]
            t = tensors[key].reshape(-1)
            mlx_key = _remap_vae_key(key)
            if mlx_key:
                out[mlx_key] = np.ascontiguousarray(t.astype(np_dtype))

        elif key.endswith(".weight"):
            consumed.add(key)
            w = tensors[key]
            if len(w.shape) == 3:
                is_transpose = ".conv_t1" in key
                w = w.transpose(1, 2, 0) if is_transpose else w.transpose(0, 2, 1)
            mlx_key = _remap_vae_key(key)
            if mlx_key:
                out[mlx_key] = np.ascontiguousarray(w.astype(np_dtype))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "vae_weights.safetensors"
    print(f"  Saving {len(out)} tensors → {out_path} …")
    save_np(out, str(out_path))
    print(f"  ✓ VAE: {len(out)} tensors written")
    return True


def copy_tokenizer(src_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer files to output LM directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "special_tokens_map.json", "added_tokens.json",
                 "chat_template.jinja", "config.json"]:
        src = src_dir / name
        if src.exists():
            dst = output_dir / ("lm_" + name)
            shutil.copy2(src, dst)
            print(f"  Copied → {dst.name}")


def inspect_file(path: Path) -> None:
    print(f"\n=== {path} ===")
    tensors = _load_safetensors_numpy(path)
    for key in sorted(tensors.keys()):
        t = tensors[key]
        print(f"  {key:80s}  {list(t.shape)}")


# ---------------------------------------------------------------------------
# End-to-end validation
# ---------------------------------------------------------------------------

def validate_output(output_dir: Path) -> bool:
    dit_path     = output_dir / "dit" / "dit_weights.safetensors"
    silence_path = output_dir / "dit" / "silence_latent.safetensors"
    lm_path      = output_dir / "lm"  / "lm_weights.safetensors"
    vae_path     = output_dir / "vae" / "vae_weights.safetensors"
    text_path    = output_dir / "text" / "text_weights.safetensors"

    ok = True
    for label, path, expected_min in [
        ("DiT", dit_path, 100),
        ("Silence latent", silence_path, 1),
        ("LM",  lm_path,  50),
        ("VAE", vae_path, 50),
        ("Text encoder", text_path, 100),
    ]:
        if not path.exists():
            print(f"  ✗ {label}: file not found at {path}", file=sys.stderr)
            ok = False
            continue
        keys = list(_load_safetensors_numpy(path).keys())
        size_mb = path.stat().st_size / 1e6
        print(f"  {label}: {len(keys)} tensors, {size_mb:.0f} MB")
        if len(keys) < expected_min:
            print(f"  ✗ {label}: only {len(keys)} tensors (expected ≥ {expected_min})", file=sys.stderr)
            ok = False
        else:
            print(f"  ✓ {label}: tensor count OK")

    # Check for critical DiT keys (ACE-Step v1.5 turbo key names)
    if dit_path.exists():
        keys = set(_load_safetensors_numpy(dit_path).keys())
        critical = [
            "decoder.layers.0.selfAttn.qProj.weight",
            "decoder.layers.0.selfAttnNorm.weight",
            "decoder.timeEmbed.linear1.weight",
            "decoder.projIn.weight",
            "decoder.projOut.weight",
            "decoder.scaleShiftTable",
            "lyricEncoder.embedTokens.weight",
            "lyricEncoder.norm.weight",
            "nullConditionEmb",
            "textProjector.weight",
            "timbreEncoder.embedTokens.weight",
            "timbreEncoder.norm.weight",
        ]
        for k in critical:
            if k in keys:
                print(f"  ✓ DiT key present: {k}")
            else:
                print(f"  ✗ DiT key MISSING: {k}", file=sys.stderr)
                ok = False

    if silence_path.exists():
        tensors = _load_safetensors_numpy(silence_path)
        latent = tensors.get("silence_latent")
        if latent is None:
            print("  ✗ Silence latent key MISSING: silence_latent", file=sys.stderr)
            ok = False
        elif len(latent.shape) == 3 and latent.shape[0] == 1 and latent.shape[2] == 64:
            print(f"  ✓ Silence latent shape OK: {list(latent.shape)}")
        else:
            print(f"  ✗ Silence latent shape invalid: {list(latent.shape)}", file=sys.stderr)
            ok = False

    # Check for critical LM keys
    if lm_path.exists():
        keys = set(_load_safetensors_numpy(lm_path).keys())
        critical_lm = [
            "embedTokens.weight",
            "layers.0.selfAttn.qProj.weight",
            "layers.0.inputLayernorm.weight",
            "norm.weight",
        ]
        for k in critical_lm:
            if k in keys:
                print(f"  ✓ LM key present: {k}")
            else:
                print(f"  ✗ LM key MISSING: {k}", file=sys.stderr)
                ok = False

    # Check for critical text-encoder keys (Qwen3-Embedding-0.6B)
    if text_path.exists():
        keys = set(_load_safetensors_numpy(text_path).keys())
        critical_text = [
            "embedTokens.weight",
            "layers.0.selfAttn.qProj.weight",
            "layers.0.selfAttn.qNorm.weight",
            "layers.0.inputLayernorm.weight",
            "norm.weight",
        ]
        for k in critical_text:
            if k in keys:
                print(f"  ✓ Text encoder key present: {k}")
            else:
                print(f"  ✗ Text encoder key MISSING: {k}", file=sys.stderr)
                ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_MODELS_DIR = Path.home() / "Library" / "Application Support" / "Auralux" / "Models"

# dit_repo:    HuggingFace repo containing the DiT checkpoint.
# dit_subdir:  Subdirectory within the repo (None = repo root).
# output_name: Local model directory and HF Team-AER repo suffix.
# sharded:     True if the DiT uses model.safetensors.index.json + shard files.
# xl:          True for XL (2560-dim decoder) checkpoints.
#
# NOTE: Verify these repo paths against https://huggingface.co/ACE-Step before
# running — ACE-Step may reorganise or rename repos between releases.
_VARIANT_CONFIG = {
    "turbo": {
        "dit_repo":    "ACE-Step/Ace-Step1.5",
        "dit_subdir":  "acestep-v15-turbo",
        "output_name": "ace-step-v1.5-mlx",
        "sharded":     False,
        "xl":          False,
    },
    "sft": {
        "dit_repo":    "ACE-Step/Ace-Step1.5",
        "dit_subdir":  "acestep-v15-sft",
        "output_name": "ace-step-v1.5-sft-mlx",
        "sharded":     False,
        "xl":          False,
    },
    "base": {
        "dit_repo":    "ACE-Step/Ace-Step1.5",
        "dit_subdir":  "acestep-v15-base",
        "output_name": "ace-step-v1.5-base-mlx",
        "sharded":     False,
        "xl":          False,
    },
    "xl-turbo": {
        "dit_repo":    "ACE-Step/Ace-Step1.5",
        "dit_subdir":  "acestep-v15-xl-turbo",
        "output_name": "ace-step-v1.5-xl-turbo-mlx",
        "sharded":     True,
        "xl":          True,
    },
    "xl-sft": {
        "dit_repo":    "ACE-Step/Ace-Step1.5",
        "dit_subdir":  "acestep-v15-xl-sft",
        "output_name": "ace-step-v1.5-xl-sft-mlx",
        "sharded":     True,
        "xl":          True,
    },
    "xl-base": {
        "dit_repo":    "ACE-Step/Ace-Step1.5",
        "dit_subdir":  "acestep-v15-xl-base",
        "output_name": "ace-step-v1.5-xl-base-mlx",
        "sharded":     True,
        "xl":          True,
    },
}


def default_output_dir(variant: str) -> Path:
    return _MODELS_DIR / _VARIANT_CONFIG[variant]["output_name"]


def _symlink_shared_components(out_dir: Path, turbo_dir: Path) -> None:
    """Symlink VAE, LM, text, and silence_latent from the turbo output dir.

    The SFT checkpoint shares all non-DiT components with turbo.
    Relative symlinks keep the directory relocatable.
    """
    for subdir in ("vae", "lm", "text"):
        src = turbo_dir / subdir
        dst = out_dir / subdir
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        if src.exists():
            rel = os.path.relpath(src, out_dir)
            os.symlink(rel, dst)
            print(f"  Symlinked {subdir}/ → {rel}")
        else:
            print(f"  WARNING: turbo {subdir}/ not found at {src} — run turbo conversion first.")

    # silence_latent lives under dit/
    src_silence = turbo_dir / "dit" / "silence_latent.safetensors"
    dst_silence = out_dir / "dit" / "silence_latent.safetensors"
    dst_silence.parent.mkdir(parents=True, exist_ok=True)
    if dst_silence.is_symlink() or dst_silence.exists():
        dst_silence.unlink()
    if src_silence.exists():
        rel = os.path.relpath(src_silence, dst_silence.parent)
        os.symlink(rel, dst_silence)
        print(f"  Symlinked dit/silence_latent.safetensors → {rel}")
    else:
        print(f"  WARNING: turbo silence_latent not found at {src_silence} — run turbo conversion first.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and convert ACE-Step v1.5 weights to MLX format"
    )
    parser.add_argument(
        "--variant",
        choices=list(_VARIANT_CONFIG.keys()),
        default="turbo",
        help="Which DiT checkpoint to convert (default: turbo). All non-turbo variants "
             "symlink shared components (VAE/LM/text/silence_latent) from the turbo "
             "output directory, so convert turbo first.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ~/Library/Application Support/Auralux/Models/<variant>/)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float16",
        help="Weight dtype (default: float16 — halves disk/memory use)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect already-downloaded source weights and exit",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate already-converted output weights and exit",
    )
    parser.add_argument(
        "--skip-dit", action="store_true", help="Skip DiT download + conversion"
    )
    parser.add_argument(
        "--skip-lm", action="store_true",
        help="Skip LM download + conversion (ignored for non-turbo variants — shared via symlink)"
    )
    parser.add_argument(
        "--skip-vae", action="store_true",
        help="Skip VAE download + conversion (ignored for non-turbo variants — shared via symlink)"
    )
    parser.add_argument(
        "--skip-text", action="store_true",
        help="Skip text-encoder download + conversion (ignored for non-turbo variants — shared via symlink)"
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "auralux-weights"),
        help="Directory for raw HuggingFace downloads (default: ~/.cache/auralux-weights)",
    )
    args = parser.parse_args()

    if not _check_deps():
        return 1

    variant    = args.variant
    vcfg       = _VARIANT_CONFIG[variant]
    out_dir    = Path(args.output_dir) if args.output_dir else default_output_dir(variant)
    turbo_dir  = _MODELS_DIR / _VARIANT_CONFIG["turbo"]["output_name"]
    cache_dir  = Path(args.cache_dir)
    token      = args.hf_token

    # ── Validate-only mode ────────────────────────────────────────────────────
    if args.validate_only:
        print(f"\nValidating {out_dir} …")
        ok = validate_output(out_dir)
        return 0 if ok else 1

    # ── Inspect-only mode ─────────────────────────────────────────────────────
    if args.inspect_only:
        dit_src = cache_dir / "dit" / "model.safetensors"
        lm_src  = cache_dir / "lm"  / "model.safetensors"
        if dit_src.exists():
            inspect_file(dit_src)
        if lm_src.exists():
            inspect_file(lm_src)
        return 0

    ok = True

    # ── DiT conversion ────────────────────────────────────────────────────────
    if not args.skip_dit:
        dit_subdir = vcfg["dit_subdir"]
        dit_repo   = vcfg["dit_repo"]
        is_sharded = vcfg["sharded"]
        dit_cache  = cache_dir / "dit" / dit_subdir

        if is_sharded:
            # Sharded: look for index file; single-file fallback for completeness.
            index_file  = dit_cache / "model.safetensors.index.json"
            single_file = dit_cache / "model.safetensors"
            needs_download = not index_file.exists() and not single_file.exists()
            if needs_download:
                print(f"\n[DiT] Downloading sharded {dit_subdir} from {dit_repo} …")
                _download_dir(
                    repo_id=dit_repo,
                    subdir=dit_subdir,
                    local_dir=dit_cache.parent,
                    token=token,
                    patterns=[f"{dit_subdir}/*"],
                )
            if index_file.exists():
                print(f"\n[DiT] Converting sharded → {out_dir}/dit/ …")
                ok &= convert_dit(dit_cache, out_dir / "dit", args.dtype, sharded=True)
            elif single_file.exists():
                print(f"\n[DiT] Converting single-file (expected sharded) → {out_dir}/dit/ …")
                ok &= convert_dit(single_file, out_dir / "dit", args.dtype, sharded=False)
            else:
                print(f"ERROR: DiT weights not found in {dit_cache}", file=sys.stderr)
                ok = False
        else:
            dit_src = dit_cache / "model.safetensors"
            if not dit_src.exists():
                print(f"\n[DiT] Downloading {dit_subdir} from {dit_repo} …")
                _download_dir(
                    repo_id=dit_repo,
                    subdir=dit_subdir,
                    local_dir=dit_cache.parent,
                    token=token,
                    patterns=[f"{dit_subdir}/*"],
                )
            if dit_src.exists():
                print(f"\n[DiT] Converting → {out_dir}/dit/ …")
                ok &= convert_dit(dit_src, out_dir / "dit", args.dtype, sharded=False)
            else:
                print(f"ERROR: DiT weights not found at {dit_src}", file=sys.stderr)
                ok = False

    # ── Shared components: convert for turbo, symlink for others ─────────────
    is_turbo = variant == "turbo"

    if is_turbo:
        # Turbo: convert silence_latent alongside the DiT weights.
        if not args.skip_dit:
            dit_subdir  = vcfg["dit_subdir"]
            silence_src = cache_dir / "dit" / dit_subdir / "silence_latent.pt"
            if silence_src.exists():
                ok &= convert_silence_latent(silence_src, out_dir / "dit", args.dtype)
            else:
                print(f"ERROR: silence_latent.pt not found at {silence_src}", file=sys.stderr)
                ok = False

        # ── LM conversion ─────────────────────────────────────────────────────
        if not args.skip_lm:
            lm_cache = cache_dir / "lm"
            lm_src   = lm_cache / "model.safetensors"

            if not lm_src.exists():
                print(f"\n[LM] Downloading from ACE-Step/acestep-5Hz-lm-0.6B …")
                _download_dir(
                    repo_id="ACE-Step/acestep-5Hz-lm-0.6B",
                    subdir=None,
                    local_dir=lm_cache,
                    token=token,
                    patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"],
                )

            if lm_src.exists():
                print(f"\n[LM] Converting → {out_dir}/lm/ …")
                ok &= convert_lm(lm_src, out_dir / "lm", args.dtype)
                copy_tokenizer(lm_cache, out_dir / "lm")
            else:
                print(f"ERROR: LM weights not found at {lm_src}", file=sys.stderr)
                ok = False

        # ── VAE conversion ────────────────────────────────────────────────────
        if not args.skip_vae:
            vae_cache = cache_dir / "vae"
            vae_src   = vae_cache / "vae" / "diffusion_pytorch_model.safetensors"

            if not vae_src.exists():
                print(f"\n[VAE] Downloading from ACE-Step/Ace-Step1.5 …")
                _download(
                    repo_id="ACE-Step/Ace-Step1.5",
                    filename="vae/diffusion_pytorch_model.safetensors",
                    local_dir=str(vae_cache),
                    token=token,
                )

            if vae_src.exists():
                print(f"\n[VAE] Converting → {out_dir}/vae/ …")
                ok &= convert_vae(vae_src, out_dir / "vae", args.dtype)
            else:
                print(f"ERROR: VAE weights not found at {vae_src}", file=sys.stderr)
                ok = False

        # ── Text encoder conversion (Qwen3-Embedding-0.6B) ───────────────────
        if not args.skip_text:
            text_cache = cache_dir / "text"
            text_src   = text_cache / "Qwen3-Embedding-0.6B" / "model.safetensors"

            if not text_src.exists():
                print(f"\n[Text] Downloading Qwen3-Embedding-0.6B from ACE-Step/Ace-Step1.5 …")
                _download_dir(
                    repo_id="ACE-Step/Ace-Step1.5",
                    subdir="Qwen3-Embedding-0.6B",
                    local_dir=text_cache,
                    token=token,
                    patterns=["Qwen3-Embedding-0.6B/*"],
                )

            if text_src.exists():
                print(f"\n[Text] Converting → {out_dir}/text/ …")
                ok &= convert_text_encoder(text_src, out_dir / "text", args.dtype)
                copy_text_tokenizer(text_src.parent, out_dir / "text")
            else:
                print(f"ERROR: Text-encoder weights not found at {text_src}", file=sys.stderr)
                ok = False

    else:
        # Non-turbo: symlink shared components from the turbo output dir.
        print(f"\n[Shared] Symlinking VAE / LM / text / silence_latent from turbo dir …")
        _symlink_shared_components(out_dir, turbo_dir)

    # ── Validate output ───────────────────────────────────────────────────────
    if ok:
        print(f"\n[Validate] Checking converted output …")
        ok = validate_output(out_dir)

    if ok:
        print(f"\n✓ All done → {out_dir}")
        if not is_turbo:
            print(f"  Shared components are symlinked from {turbo_dir}")
        print(f"  Launch Auralux and select the '{variant}' DiT variant in Settings.")
        return 0
    else:
        print(f"\n✗ Conversion had errors. See above.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
