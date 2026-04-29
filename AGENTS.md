# AGENTS.md — Auralux AI Agent Guide

This document is the orientation for AI coding agents working on Auralux. It covers architecture, conventions, file layout, and patterns to follow when making changes.

## Project Overview

Auralux is a **fully native macOS application for AI music generation**. ACE-Step v1.5 runs in-process on Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift). There is no Python backend, no HTTP server, no IPC layer at runtime. Users describe the music they want via prompts, tags, and lyrics, and the app generates it locally.

- **Platform:** macOS 26+ (Tahoe), Apple Silicon only
- **Language:** Swift 6.2 (entire app); Python is used **only** by the offline weight converter (`tools/convert_weights.py`)
- **Build system:** Swift Package Manager (no Xcode project file)
- **UI framework:** SwiftUI with `@Observable`
- **Persistence:** SwiftData (`GeneratedTrack`, `Preset`, `Tag`)
- **Inference:** ACE-Step v1.5 ported to mlx-swift (DiT 2B + DC-HiFi-GAN VAE + Qwen3 text encoder + optional 5 Hz LM 0.6B)

## Repository Structure

```
auralux/
├── Auralux/                              # SwiftUI macOS application (single binary)
│   ├── AuraluxApp.swift                  # @main; configures MLX cache, ModelContainer, env injection
│   ├── Info.plist
│   ├── Entitlements.plist                # App Sandbox + network/files/audio
│   ├── Components/                       # Reusable UI (TagChip, SliderControl, EngineStatusView, …)
│   ├── Inference/                        # Native Swift inference engine
│   │   ├── NativeInferenceEngine.swift   # @MainActor coordinator
│   │   ├── DiT/                          # DiT, samplers, VAE, audio tokenizer, weight loaders
│   │   ├── LM/                           # Optional 5 Hz audio-token LM + tokenizer
│   │   └── Text/                         # Qwen3 encoder + tokenizer
│   ├── Models/                           # SwiftData @Model + GenerationParameters + DiTVariant + GenerationMode
│   ├── Services/                         # AudioPlayer/Export, History, Preset, ModelDownloader, ModelManager, PlaybackDiagnostics
│   ├── Utilities/                        # AppLogger, Constants, AudioFFT, FileUtilities
│   ├── ViewModels/                       # @Observable state (Generation, Player, History, Settings, Sidebar)
│   ├── Views/
│   │   ├── ContentView.swift
│   │   ├── LogViewerView.swift
│   │   ├── AudioToAudio/
│   │   ├── Generation/
│   │   ├── History/
│   │   ├── Onboarding/SetupView.swift
│   │   ├── Player/
│   │   ├── Settings/
│   │   └── Sidebar/
│   └── Resources/
├── AuraluxTests/                         # Swift unit + MLX integration tests
├── docs/                                 # ARCHITECTURE, DEVELOPMENT, PENDING_PLAN, RELEASE_CHECKLIST
├── tools/convert_weights.py              # PyTorch → MLX weight converter (XL / custom variants)
├── modeling_acestep_v15_turbo.py         # Reference PyTorch model (consumed by the converter)
├── .github/workflows/ci.yml              # macos-26 swift build + CI-safe swift test + Python syntax
├── Package.swift                         # SPM manifest, Swift 6.2, mlx-swift dependency
├── README.md / CHANGELOG.md / CONTRIBUTING.md / CODE_OF_CONDUCT.md / SECURITY.md / SUPPORT.md
└── LICENSE
```

## Architecture

### Single-process model

Auralux runs as one Swift binary. The SwiftUI app and the inference engine share an address space; weight loading, sampling, VAE decode, and audio writing all happen inside the same process via mlx-swift. No subprocess is spawned at runtime.

`tools/convert_weights.py` is the **only** place Python is involved, and it runs offline from the developer's shell — the running app never invokes it.

### Layer diagram

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
│   DiT + VAE + Qwen3 + (optional) LM (mlx-swift)   │
└──────────────────────────────────────────────────┘
```

### Inference engine (`Auralux/Inference/`)

`NativeInferenceEngine` is `@MainActor @Observable`. It owns:

- The active `DiTVariant`'s weights (DiT, VAE, Qwen3 encoder, optional LM, BPE tokenizer, silence latent)
- `modelState: ModelState` — `notDownloaded`, `downloading(progress)`, `downloaded`, `loading`, `ready`, `error(String)`
- `isGenerating: Bool` plus the active generation `Task` and `AsyncThrowingStream` continuation

Heavy work (weight load, generation) is dispatched via `Task.detached(priority: .userInitiated)` so the main actor stays responsive. Generation progress is streamed back via `AsyncThrowingStream<GenerationProgress, Error>` (`preparing` / `step` / `saving` / `completed`).

`DiTVariant` (in `Auralux/Models/DiTVariant.swift`) is the source of truth for variant metadata: display name, MLX directory name, default num steps, default CFG scale, max steps, CFG-distillation flag, and `canDownloadInApp`.

Variants:

| Variant | Steps | CFG-distilled | App download |
|---------|-------|---------------|--------------|
| `turbo` | 8 | yes | yes (full bundle) |
| `sft` | 60 | no | yes (DiT-only; symlinks into turbo) |
| `base` | 60 | no | yes (DiT-only; symlinks into turbo) |
| `xl-turbo` / `xl-sft` / `xl-base` | — | — | no — `tools/convert_weights.py` |

Generation modes (`GenerationMode`): `text2music`, `cover`, `repaint`, `extract` are wired end-to-end. `text2musicLM` is reserved for the LM-driven hint path and is currently `isImplemented == false`.

### State machine

```
notDownloaded ──▶ downloading(progress) ──▶ downloaded ──▶ loading ──▶ ready
       ▲                                                                │
       └──────────────── error(message) ◀───────────────────────────────┘
```

`ContentView.task` calls `engine.checkStatus()` on first appearance. If the variant's required files are present on disk it transitions to `downloaded`, otherwise to `notDownloaded` and `engine.isOnboarding = true`, which causes `ContentView` to overlay `SetupView`.

### Generation flow

1. User configures `GenerationParameters` in `GenerationView`.
2. `GenerationViewModel` calls `NativeInferenceEngine.generate(request:)`.
3. The engine picks the sampler for the variant (`TurboSampler` for CFG-distilled, `CFGSampler` for base/SFT) and runs Qwen3 text encoding → DiT sampling → VAE decode.
4. Output is written to `~/Library/Application Support/Auralux/Generated/` and persisted to SwiftData via `HistoryService`.
5. The track appears in history and plays back in `PlayerView`.

### Model storage

- Root: `~/Library/Application Support/Auralux/Models/<DiTVariant.mlxDirectoryName>/`
- Required files per variant: `dit/dit_weights.safetensors`, `dit/silence_latent.safetensors`, `lm/lm_weights.safetensors`, `vae/vae_weights.safetensors`, `text/text_weights.safetensors`, `text/text_vocab.json`, `text/text_merges.txt`
- `ModelDownloader.manifest(for:)` is the source of truth for files, repo IDs, and sizes
- Non-turbo variants symlink `lm/`, `vae/`, `text/` into the turbo directory

### Persistence

SwiftData models:

- `GeneratedTrack` — generated music metadata, parameters, file path, favorite
- `Preset` — saved generation parameter configurations
- `Tag` — reusable tag library with usage counts

The `ModelContainer` is created once in `AuraluxApp.init()` and injected via `.modelContainer()`.

## Coding Conventions

### Swift

- **Swift 6.2** strict concurrency — `@Sendable`, `actor`, structured concurrency
- `@Observable` macro for ViewModels and the engine (not `ObservableObject`)
- `@Environment` for dependency injection (not singletons, except `AppLogger.shared`)
- `@MainActor` on ViewModels, `NativeInferenceEngine`, and any service that mutates UI state
- `actor` for thread-safe services (`ModelDownloader`)
- `async/await` everywhere; **no Combine**
- SwiftData `@Model` macro for persistence
- `AppLogger.shared` with appropriate categories — never `print()`
- All constants in `AppConstants` enum (no magic strings)
- File name matches the primary type (`FooService.swift` contains `FooService`)
- For new fields on `GenerationParameters`, default-decode in `init(from:)` so existing presets keep working

### Python (converter only)

- Python 3.11+
- `tools/convert_weights.py` is offline tooling; it must not import any Swift / app code, and the running app must not depend on it
- Validate with `python3 -m py_compile modeling_acestep_v15_turbo.py tools/convert_weights.py`

### Project patterns

- **MVVM + Services**: Views → ViewModels → Services → engine
- **Service injection** in `AuraluxApp.init()` via `.environment(...)`
- **No singletons** for business logic (only `AppLogger.shared`)
- **Graceful guard**: Generate is disabled until `modelState == .ready`. `SetupView` overlays `ContentView` whenever `engine.isOnboarding` is true.
- **MLX memory hygiene**: `MLX.Memory.cacheLimit` is capped in `AuraluxApp.init()` (1 GB; 512 MB in low-memory mode) because MLX otherwise retains every buffer it has allocated.

### Git & branch conventions

- Branches: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`
- Commits: imperative summary (`Add CFG twin-pass sampler`, not `Added...`)
- Keep commits logically grouped; prefer small, focused PRs
- CI runs `swift build`, the CI-safe `swift test` slice, and `python -m py_compile` on the converter

## Build & Run

```bash
swift build                                # Build
swift run Auralux                          # Run (first launch shows SetupView and downloads weights)
swift test                                 # Run tests (use --skip pattern below in CI)
swift test --skip 'ACEStepDiTTests|ACEStepLMTests|FeasibilityProbeTests|Qwen3ConditioningTests|Qwen3RealWeightsTests'
python3 -m py_compile modeling_acestep_v15_turbo.py tools/convert_weights.py
python tools/convert_weights.py --variant xl-turbo   # Optional: convert XL or custom weights
```

Or open `Package.swift` in Xcode and run the `Auralux` target.

## Dependencies

### Swift (Package.swift)

| Package | Imports | Purpose |
|---------|---------|---------|
| `apple/swift-collections` (1.1.0+) | `Collections` | Efficient data structures (OrderedDictionary, Deque) |
| `markiv/SwiftUI-Shimmer` (1.5.1+) | `Shimmer` | Loading shimmer effects |
| `ml-explore/mlx-swift` (0.21.0+) | `MLX`, `MLXNN`, `MLXRandom` | Inference (Metal / Apple Silicon) |

### Python (tooling only)

`tools/convert_weights.py` and `modeling_acestep_v15_turbo.py` use PyTorch + safetensors. They are not exercised at runtime. Install requirements ad-hoc; there is no project-managed virtual environment.

### System frameworks

- **SwiftUI** — UI
- **SwiftData** — persistence
- **AVFoundation / AVAudioEngine** — playback and export
- **Accelerate** — vDSP FFT for the spectrum analyzer
- **AppKit** — `AppDelegate` activation policy, log window

## Settings (UserDefaults)

`SettingsViewModel.Keys.*`:

- `settings.ditVariant` — active `DiTVariant`
- `settings.defaultMode`, `settings.defaultNumSteps`, `settings.defaultScheduleShift`, `settings.defaultCfgScale`
- `settings.useLM` — load the optional 5 Hz audio-token LM (~1.2 GB resident)
- `settings.quantizationMode` — currently `fp16` only
- `settings.lowMemoryMode` — halves `MLX.Memory.cacheLimit`

There are no environment variables for runtime configuration.

## Testing

Tests live in `AuraluxTests/`:

- **ModelTests** — SwiftData model creation and validation
- **ServiceTests** — service-layer behavior and error handling
- **ViewModelTests** — state transitions, preset application, settings persistence
- **MLX integration suites** — `ACEStepDiTTests`, `ACEStepLMTests`, `FeasibilityProbeTests`, `Qwen3ConditioningTests`, `Qwen3RealWeightsTests`. Skipped in CI; run from Xcode against real weights.

## Key Files to Know

| File | Why it matters |
|------|---------------|
| `Auralux/AuraluxApp.swift` | Entry point; MLX cache config, ModelContainer, env injection |
| `Auralux/Views/ContentView.swift` | Root view; bootstraps services and overlays `SetupView` when onboarding |
| `Auralux/Inference/NativeInferenceEngine.swift` | Engine state machine, download/load/generate orchestration |
| `Auralux/Inference/DiT/TurboSampler.swift` | 8-step CFG-distilled sampler (Turbo / XL Turbo) |
| `Auralux/Inference/DiT/CFGSampler.swift` | Twin-pass CFG sampler (base / SFT / XL base / XL SFT) |
| `Auralux/Services/ModelDownloader.swift` | HuggingFace manifest, sequential resumable downloads |
| `Auralux/Models/DiTVariant.swift` | Variant metadata (defaults, max steps, CFG-distillation flag) |
| `Auralux/Models/GenerationParameters.swift` | Generation request DTO with default-decoding |
| `Auralux/Utilities/Constants.swift` | App-wide constants |
| `Package.swift` | SPM manifest, dependency versions, target config |

## Common Tasks

### Adding a new View

1. Create `Auralux/Views/<Section>/<Name>View.swift`.
2. Use `@Environment` for ViewModels / engine.
3. Follow existing patterns in the same section.
4. Put new state on the appropriate ViewModel.

### Adding a new Service

1. Create `Auralux/Services/<Name>Service.swift`.
2. Use `@Observable` (UI-facing) or `actor` (thread-safe).
3. Instantiate in `AuraluxApp.init()` and inject via `.environment(...)`.
4. Add tests in `AuraluxTests/ServiceTests.swift`.

### Adding a new SwiftData Model

1. Create `Auralux/Models/<Name>.swift` with `@Model`.
2. Register in the `ModelContainer` initializer in `AuraluxApp.init()`.
3. Add tests in `AuraluxTests/ModelTests.swift`.

### Touching the inference engine

1. Edits go in `Auralux/Inference/`. Keep weight-loader code symmetrical with `tools/convert_weights.py` so that converted bundles round-trip.
2. Run the MLX integration suites locally in Xcode against real weights.
3. If you add a new required file to a variant bundle, update both `NativeInferenceEngine.isDownloaded(_:)` and `ModelDownloader.manifest(for:)`.

### Adding a generation knob

1. Extend `GenerationParameters` with a default value.
2. Default-decode the new key in `init(from:)` so existing presets keep loading.
3. Plumb it through `GenerationView` / `ParameterControlsView`.
4. Add a Settings default in `SettingsViewModel` if appropriate.

## Important Notes

- **No Xcode project file** — the project uses SPM exclusively (`Package.swift`).
- **Models are not in the repo** — converted MLX weights are downloaded into `~/Library/Application Support/Auralux/Models/` at runtime (Turbo / SFT / Base) or produced by `tools/convert_weights.py` (XL / custom).
- **The app runs as an SPM executable** — `AppDelegate` promotes the process to a regular GUI application (menu bar, Dock icon).
- **No Combine** — the codebase uses Swift structured concurrency (`async/await`, `Task`, `AsyncThrowingStream`).
- **No Python at runtime** — the only Python in the repo is offline conversion tooling.
