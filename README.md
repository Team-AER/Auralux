# Auralux

Auralux is a native macOS application for AI music generation. It runs ACE-Step v1.5 entirely on Apple Silicon — fully native Swift, no Python, no servers, no cloud. Generate music from text prompts and lyrics, with playback, history, and multi-format export.

## Features

- **Text-to-music generation** — describe the music you want; the app generates it locally
- **Lyrics support** — verse/chorus/bridge structure with ISO-639 language hint
- **Tag system** — genre, instrument, and mood tags
- **Multiple DiT variants** — Turbo (8-step CFG-distilled), SFT (60-step), Base (60-step)
- **Generation modes** — `text2music`, `cover`, `repaint`, `extract`
- **DiT knobs** — number of steps, schedule shift, CFG scale (where applicable)
- **Real-time playback** — waveform visualization and FFT spectrum analyzer
- **Audio export** — WAV, AAC (.m4a), or ALAC (.m4a). FLAC and MP3 aren't supported by Apple's encoder; the picker filters them out.
- **Preset system** — save and reuse generation configurations
- **Generation history** — browse, search, and favorite past tracks
- **Audio import** — drag-and-drop reference / source audio for cover, repaint, extract
- **On-device inference** — pure Swift via [mlx-swift](https://github.com/ml-explore/mlx-swift)
- **In-app model download** — first-launch onboarding fetches MLX-converted weights from HuggingFace
- **Log viewer** — built-in window for debugging and monitoring
- **Sandboxed** — App Sandbox enabled (no Python subprocess required)

## System Requirements

- macOS 26+
- Apple Silicon (M1 or later)
- 16 GB RAM recommended
- ~6 GB free disk space for the Turbo variant (more if you add SFT / Base / XL)
- Internet connection for the initial model download

## Quick Start

```bash
swift run Auralux
```

Or open `Package.swift` in Xcode and run the `Auralux` target.

On first launch the in-app onboarding panel downloads the converted MLX weights from HuggingFace (`Team-AER/ace-step-v1.5-mlx`) into `~/Library/Application Support/Auralux/Models/`. After that the app loads weights into memory on demand and generation runs entirely locally.

### Optional: convert XL or custom weights

The app downloads the Turbo, SFT, and Base variants directly. The XL variants and any custom checkpoints require a one-time conversion from the original PyTorch weights:

```bash
python tools/convert_weights.py --variant xl-turbo
```

Converted weights are written into `~/Library/Application Support/Auralux/Models/<variant>/`.

## Running Tests

```bash
swift test
```

CI runs the deterministic Model / Service / ViewModel suites. The MLX inference suites (`ACEStepDiTTests`, `ACEStepLMTests`, `FeasibilityProbeTests`, `Qwen3ConditioningTests`, `Qwen3RealWeightsTests`) require local Metal / GPU and are intended to be run from Xcode.

## Project Layout

```
Auralux/                    # SwiftUI app
├── AuraluxApp.swift          # @main, MLX cache config, ModelContainer, env injection
├── Inference/                # Native Swift inference engine
│   ├── NativeInferenceEngine.swift   # Coordinator: state, download, load, generate
│   ├── DiT/                          # ACE-Step DiT, samplers, VAE, audio tokenizer
│   ├── LM/                           # 5 Hz audio-token LM (optional)
│   └── Text/                         # Qwen3 text encoder + tokenizer
├── Views/                    # Onboarding, Generation, Player, History, Settings, Sidebar, AudioToAudio
├── ViewModels/               # @Observable state (Generation, Player, History, Settings, Sidebar)
├── Services/                 # AudioPlayer, AudioExport, History, Preset, ModelDownloader, ModelManager, PlaybackDiagnostics
├── Models/                   # SwiftData models + DiTVariant + GenerationMode + GenerationParameters
├── Components/               # TagChip, SliderControl, EngineStatusView, AudioDropZone, ProgressOverlay
└── Utilities/                # AppLogger, Constants, AudioFFT, FileUtilities

AuraluxTests/               # Unit + MLX integration tests
docs/                       # Architecture, development, release docs
tools/convert_weights.py    # PyTorch → MLX weight converter (XL / custom variants)
modeling_acestep_v15_turbo.py # Reference PyTorch model (used by the converter only)
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system design, engine state machine, generation flow
- [Development Guide](docs/DEVELOPMENT.md) — setup, workflow, project structure, troubleshooting
- [Pending Plan](docs/PENDING_PLAN.md) — remaining items
- [Release Checklist](docs/RELEASE_CHECKLIST.md) — pre-release verification
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Support](SUPPORT.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## Technology

| Layer | Technology |
|-------|-----------|
| UI | SwiftUI (macOS 26+) |
| State | `@Observable`, SwiftData |
| Inference | mlx-swift (MLX, MLXNN, MLXRandom) |
| Models | ACE-Step v1.5 DiT (2B) + Qwen3 text encoder + DC-HiFi-GAN VAE + optional 5 Hz LM (0.6B) |
| Audio | AVAudioEngine, Accelerate (vDSP FFT) |
| Export | AVFoundation (WAV / AAC / ALAC) |
| Build | Swift Package Manager (Swift 6.2) |
| CI | GitHub Actions (`swift build` + CI-safe `swift test`) |

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
