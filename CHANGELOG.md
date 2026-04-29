# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed — Architecture rewrite

- **Native Swift inference engine.** Replaced the prior Python HTTP server with a single-process, fully native Swift port of ACE-Step v1.5 running on [mlx-swift](https://github.com/ml-explore/mlx-swift). No Python, no subprocess, no IPC at runtime.
- **Removed:** `AuraluxEngine/` Python backend, `setup_env.sh`, `start_api_server_macos.sh`, `EngineService` (subprocess lifecycle), `InferenceService` (HTTP client).
- **Added:** `Auralux/Inference/` (`NativeInferenceEngine`, `DiT/`, `LM/`, `Text/`) with `ACEStepDiT`, `TurboSampler`, `CFGSampler`, `DCHiFiGANVAE`, `Qwen3EncoderModel`, `AceStepAudioTokenizer`, optional 5 Hz audio-token LM.
- **App Sandbox enabled.** With no subprocess, the App Sandbox is now on (`com.apple.security.app-sandbox` = `true`); Mac App Store distribution is no longer architecturally blocked.

### Added

- **Multiple DiT variants** — `turbo` (8-step CFG-distilled), `sft` (60-step), `base` (60-step), plus XL variants behind the converter.
- **Generation modes** — `text2music` (default), `cover`, `repaint`, `extract` wired end-to-end. `text2musicLM` reserved for the LM-driven hint path.
- **DiT knobs in the request DTO** — number of steps, schedule shift, CFG scale (with `cfgScale` ignored on CFG-distilled variants), language code.
- **`ModelDownloader` actor** — sequential, file-level resumable HuggingFace downloads with weighted progress; symlinks `lm/`, `vae/`, `text/` from non-turbo variants into the turbo bundle.
- **`ModelManagerService`** — registry of MLX artifacts (turbo / sft / base / xl-*) shown in Settings.
- **`PlaybackDiagnosticsService`** — diagnostics capture for misbehaving outputs.
- **MLX memory management** — `MLX.Memory.cacheLimit` configured in `AuraluxApp.init()` (1 GB default; 512 MB in low-memory mode) plus explicit `clearCache()` after weight load and between phases.
- **Settings additions** — DiT variant picker, default mode/steps/schedule shift/CFG scale, "Load 5 Hz LM" toggle, low-memory mode, quantization mode (currently `fp16`).
- **`tools/convert_weights.py`** — offline PyTorch → MLX weight converter for XL and custom variants. CI runs `python -m py_compile` against it and `modeling_acestep_v15_turbo.py`.
- **MLX integration test suites** — `ACEStepDiTTests`, `ACEStepLMTests`, `FeasibilityProbeTests`, `Qwen3ConditioningTests`, `Qwen3RealWeightsTests` (Xcode-only; skipped in CI).

### Carried forward

- SwiftUI macOS app built with Swift 6.2, `@Observable`, and SwiftData.
- Prompt + tag + lyric editor with structured verse/chorus/bridge markup.
- Audio playback via `AVAudioEngine` with waveform visualization and real-time FFT spectrum analyzer.
- Audio export (WAV, AAC `.m4a`, ALAC `.m4a` via `AudioExportService` + AVAssetWriter). FLAC and MP3 aren't supported by Apple's encoder and are filtered out of the picker.
- Generation history (SwiftData) with search and favorites.
- Preset system with bundle bootstrap.
- Sidebar navigation, log viewer window, engine status badge.
- Centralized logging via `AppLogger` (`OSLog`).
- CI pipeline (`macos-26`): `swift build`, CI-safe `swift test`, Python syntax check.
- Community docs: contributing guide, code of conduct, security policy, support info, issue / PR templates, Dependabot config.

### Documentation

- README, ARCHITECTURE, DEVELOPMENT, PENDING_PLAN, RELEASE_CHECKLIST, AGENTS.md updated to reflect the native Swift / mlx-swift architecture and the in-app download flow.
