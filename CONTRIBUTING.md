# Contributing to Auralux

Thanks for contributing.

## Ground Rules

- Be respectful and follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Prefer small pull requests with focused scope.
- Include tests for behavior changes where possible.
- Keep generated files and unrelated refactors out of feature PRs.

## Development Setup

1. Install Xcode 26+ (or Swift 6.2+ toolchain). Python 3.11+ is only needed if you intend to convert XL or custom weights via `tools/convert_weights.py`.
2. Clone the repository.
3. Build and run:
   ```bash
   swift run Auralux
   ```
   On first launch the in-app onboarding overlay downloads the active DiT variant's weights from HuggingFace into `~/Library/Application Support/Auralux/Models/`.
4. Run tests:
   ```bash
   swift test
   ```
   In CI, MLX integration suites are skipped (they require local Metal / GPU); they should be run from Xcode when touching `Auralux/Inference/`.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full development guide, troubleshooting, and project structure.

## Branch and Commit Conventions

- Branches: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`.
- Commits: imperative summary (`Add CFG twin-pass sampler`).
- Keep commits logically grouped.

## Pull Request Checklist

- [ ] Tests added/updated for changed behavior.
- [ ] `swift build` and the CI-safe `swift test` slice pass (`--skip 'ACEStepDiTTests|ACEStepLMTests|FeasibilityProbeTests|Qwen3ConditioningTests|Qwen3RealWeightsTests'`).
- [ ] If `Auralux/Inference/` was touched, MLX integration suites pass locally in Xcode against real weights.
- [ ] `python3 -m py_compile modeling_acestep_v15_turbo.py tools/convert_weights.py` passes (if the converter or reference model was modified).
- [ ] Docs updated for user-visible changes.
- [ ] No secrets, credentials, or personal paths are introduced.

## Coding Standards

- Follow Swift 6 concurrency patterns (`@Observable`, `actor`, `Sendable`, `async/await`).
- Use `AppLogger.shared` for logging (not `print()`).
- Constants go in `AppConstants` — avoid magic strings.
- No Combine — use Swift structured concurrency.
- See [AGENTS.md](AGENTS.md) for detailed coding conventions and patterns.

## Reporting Bugs and Requesting Features

Use GitHub Issues and include:

- Clear reproduction steps.
- Expected vs actual behavior.
- Environment details (macOS version, Swift version, chip model, RAM).
