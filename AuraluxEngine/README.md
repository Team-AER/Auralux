# Auralux Engine (Local API Scaffold)

This folder contains the local Python API server used by the macOS app during development.

## What it provides

- `GET /health`
- `POST /generate`
- `GET /jobs/<id>`
- `POST /jobs/<id>/cancel`

The current implementation is a scaffold and writes a stub `.wav` file to:

`~/Library/Application Support/Auralux/Generated`

## Run locally

```bash
cd AuraluxEngine
./start_api_server_macos.sh
```

Environment variable:

- `AURALUX_SERVER_PORT` (default: `8765`)

## Development notes

- `setup_env.sh` creates `.venv` and installs requirements.
- `requirements.txt` is intentionally minimal because the current server uses Python standard library modules.
- Replace `_run_job` in `server.py` with real model inference logic.
