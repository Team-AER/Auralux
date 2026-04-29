# Privacy Policy

**App:** Cantis
**Developer:** Prakhar Shukla
**Effective date:** 2026-04-30

## Summary

Cantis is a music-generation app that runs entirely on your Mac. It has no user accounts, no analytics, no advertising, and no telemetry. The only time the app talks to the network is the one-time download of AI model weights from HuggingFace on first launch (and again if you opt into additional model variants or add a custom model). Everything else — your prompts, lyrics, generation history, imported audio, and exported tracks — stays on your device.

## Information we do not collect

We do not collect, store, or transmit any of the following:

- Personal information (name, email address, phone number, postal address)
- Account credentials — Cantis has no sign-in or account system
- Analytics, usage statistics, or telemetry of any kind
- Crash reports sent to a remote service
- Advertising identifiers (IDFA, IDFV) or any tracking identifiers
- Contacts, calendars, photos, or other system data
- Location

Cantis does not display advertising, does not use any third-party analytics SDK, and does not share any information with data brokers.

## Information stored locally on your Mac

Cantis stores the following on your computer only. This data never leaves your device.

- **Generation history** (in SwiftData): the prompts, lyrics, tags, generation parameters (duration, variance, seed), and file paths for tracks you have generated.
- **Saved presets**: prompts, lyric templates, tags, and parameters you have saved for reuse.
- **Exported audio**: WAV, AAC, or ALAC files written to locations you choose.
- **App settings** (in `UserDefaults`): preferences such as low-memory mode and the active model.
- **Model weights**: AI model files downloaded into `~/Library/Application Support/Cantis/Models/`.
- **Custom model registry** (if you add your own models): a list of HuggingFace repository identifiers in `~/Library/Application Support/Cantis/CustomModels/registry.json`.
- **Local diagnostic logs**: written via Apple's unified logging system (`OSLog`). These remain on your Mac under your control and are never transmitted by Cantis.

## Network activity

Cantis makes network requests for one purpose only: downloading AI model weights so it can run on your device.

The network destinations are:

- `https://huggingface.co` — for model weights published at `Team-AER/ace-step-v1.5-mlx`, and (if you opt into them) `Team-AER/ace-step-v1.5-sft-mlx` and `Team-AER/ace-step-v1.5-base-mlx`.
- Any HuggingFace repository you choose to add via the "custom model" feature. If you add a custom model, Cantis fetches its weights from that HuggingFace repository.

When the app downloads model files, HuggingFace receives the standard request metadata that any web request includes — your IP address, a User-Agent string, and the requested file path — under HuggingFace's own privacy policy: <https://huggingface.co/privacy>. We (the developer of Cantis) do not operate that endpoint, do not control or receive that metadata, and do not send any Cantis-specific information along with the request. Specifically, **no prompts, lyrics, generated audio, imported audio, generation history, settings, or device identifiers are transmitted as part of the model download.**

After the model files are present on your Mac, Cantis runs entirely offline. No prompts, lyrics, audio, or generation results are ever sent to any server.

## Microphone access

Cantis requests microphone access for one feature: audio-to-audio style transfer. The system permission prompt explains this with the following text:

> "Cantis needs microphone access for audio-to-audio style transfer features."

When you use this feature, audio captured from the microphone is processed locally and in memory by the on-device model. Captured audio is not stored persistently unless you explicitly export it. It is never transmitted off your Mac.

You can decline microphone access at install time, revoke it later in **System Settings → Privacy & Security → Microphone**, or simply not use the audio-to-audio feature. Text-to-music generation does not require the microphone.

## Imported audio

Cantis lets you drag and drop audio files (WAV, AAC, ALAC, etc.) as input to the cover, repaint, and extract generation modes. Imported audio is read locally through the macOS App Sandbox's user-selected file access. It is processed on-device and is never uploaded.

## Exported audio

When you export a generated track, Cantis writes a WAV, AAC (`.m4a`), or ALAC (`.m4a`) file to the location you choose. Exports are local file writes; nothing is sent over the network.

## Third-party software

Cantis is built with the following open-source libraries. These are compiled into the app and run in-process — they do not transmit data on their own:

- [`mlx-swift`](https://github.com/ml-explore/mlx-swift) — Apple's MLX framework for on-device machine learning.
- [`swift-collections`](https://github.com/apple/swift-collections) — Standard Swift data structures.
- [`swift-numerics`](https://github.com/apple/swift-numerics) — Numeric protocols and types.
- [`SwiftUI-Shimmer`](https://github.com/markiv/SwiftUI-Shimmer) — A SwiftUI loading-shimmer effect.

The only third-party *service* Cantis interacts with is HuggingFace, and only for model-weight downloads as described above.

## Children's privacy

Cantis is not directed at children under 13 and does not knowingly collect personal information from anyone, including children. The app collects no personal information from any user.

## Your control over your data

Because all your data stays on your Mac, you control it directly:

- **Delete a single track**: remove it from the in-app history.
- **Delete all generation history**: remove the SwiftData store and exported audio under `~/Library/Application Support/Cantis/`.
- **Remove model weights**: delete `~/Library/Application Support/Cantis/Models/`.
- **Uninstall**: drag Cantis to the Trash and remove `~/Library/Application Support/Cantis/`. No remote data needs to be deleted because none was ever stored remotely.

## International users (GDPR, CCPA, and similar laws)

We do not collect, store, sell, share, or process personal information about you on any server we control. As a result:

- We do not have personal information to disclose, port, correct, or delete in response to a data-subject request — the only data exists locally on your Mac, where you already have full control.
- We do not sell or "share" personal information for behavioral advertising as those terms are defined under the CCPA / CPRA.
- We do not engage in automated decision-making or profiling.

If you have a question about how this applies to you, please reach us through the channels in the **Contact** section below.

## Security

Cantis runs in the macOS App Sandbox and requests only the entitlements it needs:

- `com.apple.security.app-sandbox` — App Sandbox (required for Mac App Store).
- `com.apple.security.files.user-selected.read-write` — read and write files you explicitly choose or drag in.
- `com.apple.security.device.audio-input` — microphone access for audio-to-audio (only when you use it).
- `com.apple.security.network.client` — outbound HTTPS requests, used solely for model-weight downloads.

If you discover a security or privacy issue, please report it privately using GitHub Security Advisories on the Cantis repository (see [SECURITY.md](SECURITY.md)) rather than a public issue.

## Changes to this policy

If this policy ever changes, the new version is committed to the Cantis repository and the **Effective date** at the top of this document is updated. The full revision history is available in git.

## Contact

Cantis is maintained by an individual developer. For privacy-related questions, please use GitHub:

- **General questions**: open an issue on the Cantis GitHub repository.
- **Security or privacy vulnerabilities**: use GitHub Security Advisories on the Cantis GitHub repository (per [SECURITY.md](SECURITY.md)). Please do not file public issues for vulnerabilities.
