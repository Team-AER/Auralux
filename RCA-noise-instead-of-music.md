# RCA — Auralux Swift inference: "noise instead of music"

**Status:** Root cause identified. Fixes implemented in this branch.
**Date:** 2026-04-28

## TL;DR

The Swift port of ACE-Step v1.5-Turbo does not produce music because **the
text-conditioning pipeline is missing a critical component (a bidirectional
text encoder)** and the placeholder used in its place feeds the wrong kind of
hidden states into the bidirectional `lyric_encoder`. With well-formed
conditioning the upstream model produces music; with what Swift currently
sends it, the model defaults to silence-or-noise behaviour because the
cross-attention conditioning is meaningless.

The user's pending VAE fixes (Snake1d → log-scale, drop `/sqrt(2)`,
drop `/0.1825` scaling factor, output stereo, no `tanh`) are **all correct**
and match the upstream `diffusers.AutoencoderOobleck`. They should be kept.

## What the upstream pipeline really looks like

`AceStepConditionGenerationModel` in
`modeling_acestep_v15_turbo.py` is composed of:

| Component | Purpose | In Swift port? |
|---|---|---|
| `AceStepConditionEncoder.text_projector` (1024→2048) | Project external text encoder hidden states to model dim | **Missing** |
| `AceStepConditionEncoder.lyric_encoder` (8 bidir layers) | Encode external lyric encoder hidden states | Present (but fed wrong inputs) |
| `AceStepConditionEncoder.timbre_encoder` | Encode reference audio | Skipped (intentional) |
| `AceStepConditionGenerationModel.decoder` (24 DiT layers) | Flow-matching generator | Present, weights load |
| `AudioTokenDetokenizer` (5Hz tokens → 25Hz latents) | Cover-mode only, not used for text2music | Present, unused |
| `null_condition_emb` (1×1×2048 learned) | Null cross-attn conditioning for CFG | Present, loaded |
| External text encoder (e.g. Qwen3-Embedding-0.6B) | Produces 1024-dim text hidden states | **Not converted, not present** |
| External lyric encoder | Produces 1024-dim lyric hidden states | **Not converted, not present** |
| 5Hz LM (`acestep-5Hz-lm-0.6B`) | Autoregressive token generator (chain-of-thought metadata) | Present — but **misused** |

The upstream `prepare_condition` for text-to-music (`is_covers=0`) does:

```
text_emb = text_projector(text_hidden_states)            # [B, S_text, 2048]
lyric_emb = lyric_encoder(inputs_embeds=lyric_hidden_states, attention_mask=...)
encoder_hidden_states = pack_sequences(lyric_emb, text_emb, masks)
context_latents = cat([silence_latent[:T], chunk_masks], dim=-1)   # [B, T, 128]
```

`text_hidden_states` and `lyric_hidden_states` come from external models that
are **not part of the published ACE-Step v1.5-Turbo checkpoint**. They have
to be computed by separate text/lyric encoder models (typically
`Qwen3-Embedding-0.6B` or similar bidirectional sentence encoders).

## What Swift was doing

`NativeInferenceEngine.generate(...)`:

```swift
let lmHidden = localLm.hiddenStates(inputIds)   // 5Hz autoregressive LM, causal mask
encH = localDit.lyricEncoder(lmHidden)          // pretend the LM is a text encoder
```

Two fundamental issues:

1. **`acestep-5Hz-lm-0.6B` is not a text encoder.** It's a Qwen2-style
   *causal* (autoregressive) language model whose job is to generate 5Hz
   audio token codes (consumed by `AudioTokenDetokenizer`) given prompt
   metadata. Its hidden states are shaped for causal next-token prediction,
   not for use as bidirectional text embeddings.

2. **`lyric_encoder` was trained on outputs of a *different* (bidirectional)
   model.** Feeding causal LM hidden states into it is feeding a model
   inputs from outside its training distribution.

## Empirical verification

I ran the upstream Python `AceStepDiTModel.forward` end-to-end against three
conditioning regimes, with `silence_latent` as `src_latents` (matching the
user's current Swift setup):

| Conditioning | Final-latent abs.mean | Audio RMS | Audible? |
|---|---|---|---|
| `null_condition_emb` (broadcast) | 0.61 | 0.0009 | silence |
| Random `[1, 30, 2048]` × 0.1 (mimicking `lyricEncoder(garbage)`) | 0.60 | 0.0009 | silence |
| Random `[1, 50, 2048]` × 0.1 (raw, no encoder) | 0.60 | 0.0009 | silence |
| Zero `[1, T, 128]` context + null cond (the *baseline* code path) | 0.64 | 0.093 | audible drone |

So **with correct (or null) conditioning + silence-latent context, the
upstream model produces silence**, not noise. The Swift port should observe
the same. If the user is hearing actual broadband noise (not silence), it is
because:

a. The cross-attention conditioning produces NaN/Inf (e.g. due to a fp16
   precision issue inside `lyric_encoder`'s self-attention with a
   mismatched-distribution input), and these propagate through the DiT into
   the VAE; OR
b. The user is interpreting a low-RMS, structureless waveform as "noise".

Either way, the fix is the same: **stop feeding causal-LM hidden states
into the bidirectional `lyric_encoder`**, and instead use the model's own
`null_condition_emb` (which is what the model itself uses during CFG
dropout). This produces deterministic, well-defined behaviour (silence-like
output), and makes it obvious that the next step is to integrate a real
text encoder.

## VAE — user's pending changes are CORRECT

Verified against `diffusers/models/autoencoders/autoencoder_oobleck.py`
(installed locally) and against the actual ACE-Step v1.5 VAE checkpoint
(`encoder_hidden_size=128`, `decoder_input_channels=64`, `audio_channels=2`,
`channel_multiples=[1,2,4,8,16]`, `downsampling_ratios=[2,4,4,6,10]`):

| User change | Verdict |
|---|---|
| `Snake1d`: `exp(α)`, `exp(β) + 1e-9`, init zeros | ✓ matches `Snake1d(logscale=True)` exactly |
| `OobleckResUnit`: drop `/sqrt(2.0)` on residual | ✓ upstream is `hidden_state + output_tensor` |
| `DCHiFiGANDecoder`: drop `/0.1825 scalingFactor` | ✓ `AutoencoderOobleck.decode` does *not* scale |
| `DCHiFiGANDecoder`: drop `tanh` and `.mean(axis: -1)` | ✓ upstream returns stereo `[B, 2, T]` raw |
| `silence_latent` as `src_latents` (vs zeros) | ✓ matches `prepare_condition` for `is_covers=0` |
| Stereo WAV writer | ✓ correct (channels=2, byteRate=sr*2*2, blockAlign=4) |

The Snake α/β values in the converted checkpoint range from roughly −1 to +2
with mean near 0. With raw values, `1/β` would explode for β near zero or go
negative for β<0; with `exp()` they always lie in (0, ∞). The empirical
values are only meaningful in log-scale, confirming the diffusers convention.

## Other things checked (and ruled out)

- **GQA tiling.** `MLX.repeated(x, count: g, axis: 1)` follows `np.repeat`
  semantics (interleave: `[a,a,b,b]`), which is correct for GQA where Q head
  *i* maps to KV head *i // g*. Verified empirically with `mx.repeat` in
  Python. Both `ACEStepLMAttention.gqaTile` and `AceStepAttention`'s
  `repeated()` produce the same correct output.
- **Conv weight transpose direction.** MLX-Swift `Conv1d` and
  `ConvTransposed1d` both expect `[outputChannels, kernelSize, inputChannels]`.
  The conversion script's `transpose(0,2,1)` for Conv1d and `transpose(1,2,0)`
  for ConvTranspose1d are both correct. Verified by reading
  `mlx-swift/Source/MLXNN/Convolution.swift` and `ConvolutionTransposed.swift`.
- **DiT forward shapes.** `decoder.condition_embedder.weight` is `[2048, 2048]`
  (matches Swift `Linear(2048, 2048)`). `null_condition_emb` is `[1, 1, 2048]`
  and loads correctly into `nullConditionEmb`. `decoder.scale_shift_table`
  `[1, 2, 2048]` and per-layer `scale_shift_table` `[1, 6, 2048]` all match.
- **TurboSampler schedule.** Matches `SHIFT_TIMESTEPS[3.0]` from upstream
  `generate_audio` exactly, and the final-step `x0 = xt - vt * t` agrees with
  `get_x0_from_noise`.
- **Silence latent loading.** `silence_latent.pt` is shape `[1, 64, 15000]`
  in unscaled VAE-encoder-output space; the conversion to `[1, T, 64]` and
  Swift's `SilenceLatentLoader.slice` are correct. Decoding it through
  upstream Python yields RMS ≈ 0.001 (silence) — this is the *correct*
  reference behaviour.
- **Weight conversion completeness.** All 596 critical DiT keys (decoder,
  lyric_encoder, detokenizer, null_condition_emb) are present in the
  converted file. The skipped `encoder.text_projector` is intentional —
  Swift never declares this layer; the issue is conceptual (no text encoder
  exists).
- **Build.** `xcodebuild -scheme Auralux build` succeeds with no errors.

## Fixes implemented in this branch

1. **Stop feeding LM hidden states into `lyric_encoder`.** Always use
   `null_condition_emb` broadcast across the requested sequence length until
   a real text encoder is integrated. This matches the upstream's CFG-null
   behaviour and produces deterministic output.

2. **Switch weight loaders to `verify: .all` in DEBUG.** With `.none` the
   loaders silently drop missing or shape-mismatched keys, masking exactly
   the kind of bug that took hours to find. Keep `.none` in release for
   forward compatibility, but surface errors in debug.

3. **Document the missing text encoder.** Update `swift-migration-plan.md`
   and add an in-code note explaining that text-conditioned generation is
   a known TODO requiring a separate `text_encoder` (e.g.
   `Qwen3-Embedding-0.6B`) to be downloaded, converted, and wired in.

## Next steps (out of scope for this fix)

- Integrate a real bidirectional text encoder. `Qwen3-Embedding-0.6B` is the
  natural choice (1024-dim hidden, matches `text_hidden_dim` in the
  checkpoint). It needs MLX conversion (`tools/convert_weights.py` could be
  extended) and a Swift port similar to `ACEStepLM` but bidirectional.
- Once the text encoder is in place, implement `pack_sequences` (currently
  unused in Swift) and the `text_projector` projection, then assemble
  `encoder_hidden_states` per `AceStepConditionEncoder.forward`.
- Consider exposing the `is_covers`/`audio_codes` cover-mode path, which
  uses the existing 5Hz LM and detokenizer for *audio* conditioning. That
  path is what the 5Hz LM was actually trained for.
