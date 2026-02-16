# Contributing to Auralux

Thanks for contributing.

## Ground rules

- Be respectful and follow the Code of Conduct.
- Prefer small pull requests with focused scope.
- Include tests for behavior changes where possible.
- Keep generated files and unrelated refactors out of feature PRs.

## Development setup

1. Install Xcode 16+ (or Swift 6.0+) and Python 3.10+.
2. Clone the repository.
3. Start the local API server:
   ```bash
   cd AuraluxEngine
   ./start_api_server_macos.sh
   ```
4. In a separate terminal, run tests:
   ```bash
   swift test
   ```

## Branch and commit conventions

- Branches: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`.
- Commits: imperative summary (`Add queue retry backoff`).
- Keep commits logically grouped.

## Pull request checklist

- [ ] Tests added/updated for changed behavior.
- [ ] `swift test` passes.
- [ ] Docs updated for user-visible changes.
- [ ] No secrets, credentials, or personal paths are introduced.

## Reporting bugs and requesting features

Use GitHub Issues and include:

- Clear reproduction steps.
- Expected vs actual behavior.
- Environment details (macOS version, Swift version, CPU architecture).
