# Auralux

Auralux is an open source macOS SwiftUI application scaffold for local AI music generation.
It combines a native desktop client with a lightweight local Python API server used for generation job orchestration.

## Project status

Auralux is currently in early-stage development. The current implementation focuses on architecture, workflow wiring, and local development ergonomics.

## Current capabilities

- Native macOS app shell built with SwiftUI + SwiftData.
- Generation workflow (submit, poll, cancel) through a local HTTP bridge.
- Queueing, preset management, and history tracking.
- Basic playback/export scaffolding.
- Local Python API server scaffold for `/health`, `/generate`, `/jobs/:id`, and cancellation.
- Unit tests for core model, queue, and view-model behavior.

## System requirements

- macOS 15+
- Xcode 16+ (recommended) or Swift 6+
- Python 3.10+

## Quick start

1. Clone the repository.
2. Start the local engine server:

```bash
cd AuraluxEngine
./start_api_server_macos.sh
```

3. In another terminal, run tests:

```bash
cd /path/to/auralux
swift test
```

4. Open/build the app:

- Open the package in Xcode and run the `Auralux` executable target, or
- Run `swift run Auralux` from repository root.

## Project layout

- `Auralux/` SwiftUI app source.
- `AuraluxEngine/` local Python API server scaffold.
- `AuraluxTests/` unit tests.
- `docs/` architecture, development, and release guidance.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Release Checklist](docs/RELEASE_CHECKLIST.md)
- [Engine README](AuraluxEngine/README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Support](SUPPORT.md)

## Community and support

- Bugs and feature requests: open a GitHub issue.
- Security issues: follow `SECURITY.md` and report privately.

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
