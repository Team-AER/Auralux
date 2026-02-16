# Architecture

## High-level components

- `Auralux/` (Swift): UI, state, and local persistence.
- `AuraluxEngine/` (Python): local HTTP service for generation job orchestration.
- `AuraluxTests/` (Swift tests): behavior checks for key app components.

## Swift app structure

- `Views/`: screens and composition.
- `ViewModels/`: user-intent orchestration and state transitions.
- `Services/`: boundary logic for inference, persistence, queueing, and playback/export.
- `Models/`: app domain and SwiftData models.
- `Utilities/`: app constants and reusable helpers.

## Generation flow

1. User configures prompt/tags/parameters.
2. `GenerationViewModel` builds a `GenerationRequest` and calls `InferenceService.generate`.
3. `InferenceService` starts the local engine server if needed, submits job, and polls status.
4. On completion, a `GeneratedTrack` is stored through `HistoryService`.
5. Track appears in history and can be previewed in player views.

## Persistence

SwiftData stores:

- `GeneratedTrack`
- `Preset`
- `Tag`

## API contract (current)

- `GET /health` => service liveness.
- `POST /generate` => enqueue generation and return `jobID`.
- `GET /jobs/<id>` => status/progress/audio path.
- `POST /jobs/<id>/cancel` => request cancellation.

## Current limitations

- Engine is scaffold-only and does not run real model inference yet.
- Audio export is currently placeholder behavior.
- Dependency integration for end-to-end model execution is pending.
