# Architecture

## Overview

Auralux is a single-process, fully native macOS application. The SwiftUI app and the inference engine run in the same Swift binary; there is no Python backend, no IPC layer, and no HTTP server. Inference runs on Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift).

```
┌──────────────────────────────────────────────────┐
│           SwiftUI Views + ViewModels              │
│    (@Observable, @Environment, @Query)            │
├──────────────────────────────────────────────────┤
│              Service Layer                        │
│  AudioPlayerService · AudioExportService          │
│  HistoryService · PresetService                   │
│  ModelDownloader (actor) · ModelManagerService    │
│  PlaybackDiagnosticsService                       │
├──────────────────────────────────────────────────┤
│           SwiftData Persistence                   │
│  GeneratedTrack · Preset · Tag                    │
├──────────────────────────────────────────────────┤
│            NativeInferenceEngine                  │
│   ACE-Step v1.5 DiT + VAE + Qwen3 + (LM)          │
│            mlx-swift (Metal / ANE)                │
└──────────────────────────────────────────────────┘
```

## High-Level Components

- **`Auralux/`** (Swift) — UI, state management, audio playback/export, persistence, and the entire inference stack.
- **`Auralux/Inference/`** — native Swift port of the ACE-Step v1.5 pipeline running on mlx-swift.
- **`AuraluxTests/`** — unit tests plus MLX integration suites (Xcode-only).
- **`tools/convert_weights.py`** — one-shot PyTorch → MLX weight converter. Used to produce the XL variants and any custom checkpoints that aren't already published as MLX safetensors.

## Swift App Structure

### Views

- `Onboarding/SetupView` — first-run overlay that downloads + loads model weights
- `Generation/` — main generation UI (`GenerationView`, `LyricEditorView`, `ParameterControlsView`, `TagEditorView`)
- `Player/` — playback with `WaveformView` and `SpectrumAnalyzerView`
- `History/` — browse, search, favorite past tracks
- `Settings/` — `SettingsView`, `ModelSettingsView`
- `Sidebar/` — navigation, presets, recent tracks
- `AudioToAudio/` — audio import (cover / repaint / extract sources) and LoRA management
- `LogViewerView` — opened via `Window("Auralux Logs", id: "log-viewer")`

### ViewModels

`@Observable`, `@MainActor`:

- `GenerationViewModel` — request building, progress streaming, last track
- `PlayerViewModel` — playback state
- `HistoryViewModel` — history browsing
- `SidebarViewModel` — navigation
- `SettingsViewModel` — UserDefaults-backed settings (DiT variant, default mode/steps/CFG, LM toggle, low-memory mode)

### Services

Injected via `@Environment` from `AuraluxApp.init()`:

- **`AudioPlayerService`** — `AVAudioEngine` wrapper, waveform data, real-time FFT
- **`AudioExportService`** — audio export via `AVAssetWriter`. WAV is a passthrough copy; AAC and ALAC are transcoded to `.m4a`. FLAC and MP3 are intentionally rejected with `AudioExportError.unsupported` because `AVAssetWriter` does not support them.
- **`HistoryService`** — SwiftData CRUD for `GeneratedTrack`, search, favorites, orphan reconciliation
- **`PresetService`** — preset CRUD, bundle bootstrap
- **`ModelDownloader`** (`actor`) — sequential, resumable HuggingFace downloads with weighted progress, post-download symlinking for variants that share components
- **`ModelManagerService`** — registry of MLX artifacts (turbo / sft / base / xl-*)
- **`PlaybackDiagnosticsService`** — diagnostics capture for misbehaving outputs

### Models

SwiftData `@Model` types and request DTOs:

- **`GeneratedTrack`** — generated music metadata, parameters, file path, favorite flag
- **`Preset`** — saved generation configurations
- **`Tag`** — reusable tag library with usage tracking
- **`GenerationParameters`** — `Codable` request struct (prompt, lyrics, tags, duration, variance, seed, language, mode, numSteps, scheduleShift, cfgScale, source/refer audio, repaint mask)
- **`DiTVariant`** — `turbo` / `sft` / `base` / `xl-turbo` / `xl-sft` / `xl-base`; encodes display name, MLX directory name, default num steps and CFG scale, max steps, CFG-distillation flag, and "can download in app" flag
- **`GenerationMode`** — `text2music` / `text2musicLM` / `cover` / `repaint` / `extract`

### Components

- `EngineStatusView` — toolbar badge driven by `NativeInferenceEngine.modelState`
- `TagChip`, `SliderControl`, `ProgressOverlay`, `AudioDropZone`

### Utilities

- `AppLogger` — `OSLog` wrapper with categories (`.app`, `.inference`, `.audio`, …)
- `AudioFFT` — vDSP FFT for the spectrum analyzer
- `Constants` — `AppConstants` (app name, window dimensions, suggested tags, model directory names, HF repo IDs)
- `FileUtilities` — Application Support path helpers

## Inference Engine

`Auralux/Inference/NativeInferenceEngine.swift` is the central coordinator. It is `@MainActor @Observable`, owns weights for the active `DiTVariant`, and exposes a single `generate(request:)` API that returns an `AsyncThrowingStream<GenerationProgress, Error>`.

### Sub-components

```
Inference/
├── NativeInferenceEngine.swift     # Coordinator (download/load/generate)
├── DiT/
│   ├── ACEStepDiT.swift              # ACE-Step DiT block (2B params, MLX)
│   ├── DiTWeightLoader.swift         # safetensors → ACEStepDiT
│   ├── TurboSampler.swift            # 8-step CFG-distilled sampler
│   ├── CFGSampler.swift              # 60-step twin-pass CFG sampler (base/SFT)
│   ├── AudioVAE.swift                # DC-HiFi-GAN VAE (latent → audio)
│   ├── VAEWeightLoader.swift
│   ├── AceStepAudioTokenizer.swift   # FSQ audio token codec
│   ├── AudioFileLoader.swift         # cover / repaint / extract source loading
│   └── SilenceLatentLoader.swift     # Initial silence latent for unconditional padding
├── LM/                                # Optional 5 Hz audio-token LM (0.6B)
│   ├── ACEStepLM.swift
│   ├── ACEStepLMSampler.swift
│   ├── BPETokenizer.swift
│   └── LMWeightLoader.swift
└── Text/                              # Qwen3 text conditioning encoder
    ├── Qwen3Encoder.swift
    ├── Qwen3EncoderWeightLoader.swift
    ├── Qwen3Tokenizer.swift
    └── PackSequences.swift
```

### Variants

| Variant | Steps | CFG-distilled | App download | Notes |
|---------|-------|---------------|--------------|-------|
| `turbo` | 8 (≤20) | yes | yes | Default; ships full bundle (DiT + LM + VAE + text) |
| `sft` | 60 (≤100) | no | yes | DiT-only; symlinks `lm/`, `vae/`, `text/` from turbo |
| `base` | 60 (≤100) | no | yes | DiT-only; symlinks shared components |
| `xl-turbo` / `xl-sft` / `xl-base` | — | — | no | Require `tools/convert_weights.py` |

CFG-distilled variants (Turbo) ignore `cfgScale > 1`. Base / SFT use a twin-pass CFG sampler.

### Generation Modes

`GenerationMode` mirrors the upstream ACE-Step v1.5 feature matrix:

- `text2music` — pure text conditioning (default)
- `cover` — refer audio + source audio
- `repaint` — source audio + masked time ranges with crossfade and injection ratio
- `extract` — refer audio
- `text2musicLM` — opt-in path requiring the 5 Hz audio-token LM (not yet wired end-to-end)

## Engine State Machine

`NativeInferenceEngine.modelState` drives onboarding and the generation guard.

```
notDownloaded ──▶ downloading(progress) ──▶ downloaded ──▶ loading ──▶ ready
       ▲                                                                │
       └──────────────── error(message) ◀───────────────────────────────┘
```

| State | Meaning | UI Effect |
|-------|---------|-----------|
| `notDownloaded` | No converted weights for current variant | `SetupView` overlay shown |
| `downloading(progress)` | `ModelDownloader` is fetching the variant's manifest | Setup progress bar, engine badge yellow |
| `downloaded` | Weights on disk, not yet loaded into memory | Generate disabled until `loadModels()` |
| `loading` | Weights being read into MLX arrays on a detached task | Engine badge yellow |
| `ready` | DiT (+ optional LM) + VAE + Qwen3 resident | Engine badge green, generate enabled |
| `error(message)` | Download or load failure | Error UI with retry |

`ContentView` calls `engine.checkStatus()` on first appearance. If the variant's required files are present on disk it transitions to `downloaded`, otherwise to `notDownloaded` and shows `SetupView`. `SetupView.downloadAndLoad()` drives the variant download + load.

`isGenerating` is a separate `@Observable` flag tracked by the engine while a generation task is running.

## Concurrency Model

- **`@MainActor`** — ViewModels, `NativeInferenceEngine`
- **`actor`** — `ModelDownloader` (sequential file downloads with weighted progress callbacks)
- **`async/await`** — all async APIs use Swift structured concurrency
- **`Task.detached(priority: .userInitiated)`** — heavy work (weight loading, generation) is dispatched off the main actor
- **`AsyncThrowingStream`** — generation progress is streamed back to the ViewModel
- **No Combine**

## Generation Flow

1. User configures `GenerationParameters` in `GenerationView` (mode, prompt, tags, lyrics, duration, variance, seed, num steps, schedule shift, CFG scale, optional source/refer audio, repaint mask).
2. `GenerationViewModel` calls `NativeInferenceEngine.generate(request:)`.
3. The engine validates state, picks the sampler for the variant (`TurboSampler` for CFG-distilled, `CFGSampler` for base/SFT), and runs:
   - **Qwen3 text encoder** → conditioning tokens
   - **AceStepAudioTokenizer / AudioFileLoader** when the mode requires source/refer audio
   - **DiT sampler** → latent of shape `[B, C, T_latent]`
   - **DC-HiFi-GAN VAE** → 48 kHz audio
4. Progress is streamed via `AsyncThrowingStream<GenerationProgress, Error>` (`preparing` → `step(current, total)` → `saving` → `completed(audioURL)`).
5. The output is written to `~/Library/Application Support/Auralux/Generated/` and persisted to SwiftData via `HistoryService`.
6. The track appears in history and can be played back in `PlayerView`.

## Model Storage and Downloads

- Root: `~/Library/Application Support/Auralux/Models/<variant-directory>/`
- Variant directory names come from `DiTVariant.mlxDirectoryName` (e.g. `ace-step-v1.5-mlx`, `ace-step-v1.5-sft-mlx`).
- Each variant must contain:
  ```
  dit/dit_weights.safetensors
  dit/silence_latent.safetensors
  lm/lm_weights.safetensors
  vae/vae_weights.safetensors
  text/text_weights.safetensors
  text/text_vocab.json
  text/text_merges.txt
  ```
  Plus tokenizer JSON / merges files for the LM when `useLM` is on.
- `ModelDownloader.manifest(for:)` is the source of truth for files and approximate sizes (used to weight overall download progress).
- Non-turbo variants only ship `dit/` files in their HF repo. Their `lm/`, `vae/`, and `text/` directories are created as symlinks pointing at the turbo directory after download.
- Downloads are sequential, file-level resumable (already-present files are skipped), and emit a single weighted progress value in `[0, 1]`.

## Persistence

SwiftData stores all structured data:

- **`GeneratedTrack`** — generated music metadata and audio file paths (relative for sandbox resilience)
- **`Preset`** — saved generation parameter configurations
- **`Tag`** — reusable tag library with usage counts

Audio files are stored in `~/Library/Application Support/Auralux/Generated/`. The `ModelContainer` is created once in `AuraluxApp.init()` and injected via `.modelContainer()`.

## App Entry Point

`AuraluxApp.swift` is the composition root:

1. Caps the MLX freed-buffer pool (`MLX.Memory.cacheLimit`) at 1 GB (512 MB in low-memory mode). MLX otherwise retains every buffer it has allocated, growing resident memory to the high-water mark of the union of all phases (weight load + DiT activations + VAE decode).
2. Constructs `NativeInferenceEngine` and the ViewModels (`Generation`, `History`, `Player`, `Settings`, `Sidebar`).
3. Initializes the SwiftData `ModelContainer` for `GeneratedTrack`, `Preset`, `Tag`.
4. Injects everything into the SwiftUI environment.
5. Defines the main `WindowGroup` and the `Window("Auralux Logs", id: "log-viewer")` log viewer.
6. Includes an `AppDelegate` that promotes the SPM executable to a regular GUI application (menu bar, Dock icon) and forwards `applicationWillTerminate` to `engine.shutdown()` and the player service.

## Sandboxing and Distribution

- The app ships with the App Sandbox enabled (`com.apple.security.app-sandbox` = `true`).
- Required entitlements: `com.apple.security.network.client` (for HuggingFace downloads), `com.apple.security.files.user-selected.read-write` (for audio import / export panels), `com.apple.security.device.audio-input` (reserved for future capture features).
- Because there is no Python subprocess, sandbox compatibility is no longer an architectural blocker — Mac App Store distribution is feasible once code signing and packaging are in place. The remaining work is signing, notarization, asset design, and packaging (see `docs/PENDING_PLAN.md`).
