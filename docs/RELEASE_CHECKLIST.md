# Release Checklist

## Code and Quality

- [ ] `swift build` succeeds with no warnings.
- [ ] CI-safe `swift test --skip 'ACEStepDiTTests|ACEStepLMTests|FeasibilityProbeTests|Qwen3ConditioningTests|Qwen3RealWeightsTests'` passes on `main`.
- [ ] Full `swift test` (including MLX integration suites) passes locally in Xcode against real weights.
- [ ] `python3 -m py_compile modeling_acestep_v15_turbo.py tools/convert_weights.py` passes.
- [ ] No debug code, private paths, or credentials committed.
- [ ] CI pipeline passes (GitHub Actions).

## Engine and Inference

- [ ] First-launch onboarding (`SetupView`) downloads the Turbo variant cleanly on a fresh `~/Library/Application Support/Auralux/Models/` directory.
- [ ] Variant switching downloads the SFT and Base bundles and correctly creates symlinks into the Turbo directory for `lm/`, `vae/`, and `text/`.
- [ ] `tools/convert_weights.py --variant xl-turbo` produces a directory the engine can load.
- [ ] End-to-end text-to-music generation succeeds in each implemented mode (`text2music`, `cover`, `repaint`, `extract`).
- [ ] CFG knob is applied for `base` / `sft` and ignored for `turbo` / `xl-turbo` (CFG-distilled).
- [ ] Toggling Settings → "Load 5 Hz LM" requires a model reload but does not leak memory.
- [ ] Engine shuts down cleanly on app quit (no orphaned MLX allocations or file handles).

## Application

- [ ] Onboarding overlay disappears once `modelState` reaches `.ready`.
- [ ] `EngineStatusView` toolbar badge reflects correct state.
- [ ] Generation, playback, export (WAV / AAC / ALAC), history, and presets all functional.
- [ ] Log viewer window opens via Window menu and displays logs correctly.
- [ ] App appears in Dock and gets keyboard focus (AppDelegate `setActivationPolicy(.regular)`).
- [ ] Low-memory mode toggle takes effect on next launch (halves `MLX.Memory.cacheLimit`).

## Sandbox and Distribution

- [ ] App Sandbox enabled (`com.apple.security.app-sandbox` = `true`).
- [ ] Network entitlement allows HuggingFace downloads.
- [ ] User-selected file entitlement covers audio import and export panels.
- [ ] Hardened Runtime enabled for direct distribution.
- [ ] Notarization succeeds for the signed build.

## Documentation

- [ ] `README.md` reflects current features, requirements, and quick-start.
- [ ] `CHANGELOG.md` updated with notable changes.
- [ ] `AGENTS.md` is current with codebase structure and conventions.
- [ ] New settings or `DiTVariant` options documented in `DEVELOPMENT.md`.
- [ ] `docs/ARCHITECTURE.md` matches the current state machine and inference layout.

## Community and Governance

- [ ] `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` present and current.
- [ ] Issue and PR templates remain aligned with the workflow.

## Release Prep

- [ ] Tag/version prepared.
- [ ] Release notes summarize user-visible changes and known limitations.
- [ ] App version string updated in `AuraluxApp.swift` (`applicationVersion`).
