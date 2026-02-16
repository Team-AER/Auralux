# Development Guide

## Prerequisites

- macOS 15+
- Xcode 16+ or Swift 6+
- Python 3.10+

## First-time setup

```bash
git clone <repo-url>
cd auralux
cd AuraluxEngine
./setup_env.sh
```

## Running the engine server

```bash
cd AuraluxEngine
./start_api_server_macos.sh
```

By default the server binds to `127.0.0.1:8765`.

## Running tests

```bash
cd /path/to/auralux
swift test
```

## Suggested local checks before opening a PR

```bash
swift test
python3 -m py_compile AuraluxEngine/server.py
```

## Troubleshooting

- `serverScriptMissing`: run from repository root or ensure `AuraluxEngine/start_api_server_macos.sh` exists.
- Port conflict on `8765`: set `AURALUX_SERVER_PORT=<new-port>` before starting server.
- Python venv issues: delete `AuraluxEngine/.venv` and rerun `./setup_env.sh`.

## Coding expectations

- Keep PRs scoped and testable.
- Add or update tests for behavior changes.
- Update docs when introducing workflow or API changes.
