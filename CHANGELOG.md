# Changelog

Record completed, user-visible changes here. Open work lives in [TODO.md](TODO.md).
Measurements belong to [README.md](README.md), which owns current behaviour, so an entry
links a measurement instead of restating one. The rules are in
[AGENTS.md](AGENTS.md#changelog-rules-changelogmd).

Earlier release history has not been reconstructed. This register starts on 2026-09-13, the
day [TODO.md](TODO.md) was added, and it also covers the three changes merged just before it.

## [Unreleased]

### Added

- **Validation backlog:** [TODO.md](TODO.md) records findings and acceptance checks; this changelog records completed work.
  [AGENTS.md](AGENTS.md#known-live-defects-do-not-fix-unilaterally) points to the backlog (#14).
- **Document checks:** changelog rules, release-heading and artifact-reference validation, and CI for all test suites.
  Historical changes are backfilled and the unavailable debug-artifact location is stated explicitly (#15).

### Fixed

- **Verification:** tighter derived-number grounding, unit parsing and per-agent reporting. **Schema:** attribution and offsets added; debate violations included.
  **Cost:** paid refresh of tracked runs; rubric verdicts unchanged. See the [verified runs table](README.md#verified-runs-2026-09-13-current-code) (#11).
- **Agent protocol:** output caps are enforced, resumed outputs re-audited, and debate evidence paths checked.
  **Cost:** word overruns can trigger a corrective call; no tracked-run refresh. See [prompt bounding](README.md#prompt-bounding-is-part-of-the-protocol) (#12).
- **Resume guard:** cross-ticker files are refused and model differences reported.
  See [failure mode 12](README.md#known-failure-modes-already-handled) (#13).
