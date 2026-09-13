# Changelog

Record completed, user-visible changes here. Open work lives in [TODO.md](TODO.md).
Measurements belong to [README.md](README.md), which owns current behaviour, so an entry
links a measurement instead of restating one. The rules are in
[AGENTS.md](AGENTS.md#changelog-rules-changelogmd).

Earlier release history has not been reconstructed. This register starts on 2026-09-13, the
day [TODO.md](TODO.md) was added, and it also covers the three changes merged just before it.

## [Unreleased]

### Added

- **Validation backlog:** [TODO.md](TODO.md) tracks the open findings from the 2026-09-13
  review, with reproduction evidence, the decision each one needs and acceptance checks (#14).
- **Change tracking:** this changelog, so completed work has one dated register (#14).
- **Agent protocol:** a bull argument or bear attack must carry an evidence key path, and a
  missing or malformed one is a violation rather than a silent skip (#12).
- **Resume guard:** `--resume` refuses a file written for another ticker, and notes a stored
  model that differs from the one being called (#13).

### Changed

- **Verifier attribution:** a number is matched to the metric named near it, read with B and T
  units and written dates, and a derived value needs both operands named, so invented and
  coincidental numbers stop passing. **Schema:** `verification.unverified_by_agent` is new,
  `verification.agent_violations` includes the debate agents, and rows carry an offset.
  **Cost:** the three tracked runs were re-run with paid calls, and no rubric verdict moved.
  Rates and counts live in the
  [verified runs table](README.md#verified-runs-2026-09-13-current-code) (#11).
- **Output limits:** the stated word and item limits are enforced in code. Item overruns are
  trimmed, a word overrun earns one corrective re-call, the cleanest attempt wins, and a
  resumed result is re-audited instead of trusted. **Cost:** none, the tracked runs were
  inside every limit (#12).
- **Defect register:**
  [AGENTS.md](AGENTS.md#known-live-defects-do-not-fix-unilaterally) now points at
  [TODO.md](TODO.md) instead of reporting that nothing is tracked (#14).

### Fixed

- **Verification:** an invented paragraph of twelve numbers is rejected instead of accepted,
  and a random number between 0.05 and 100 is far less likely to pass. See
  [failure mode 10](README.md#known-failure-modes-already-handled) (#11).
- **Protocol:** the output limits were prompt text only, so a bull case with twelve arguments
  and summaries over the stated word cap shipped as compliant. See failure mode 11 (#12).
- **Orchestration:** `--resume` imported another ticker's analysis under this symbol, and the
  verifier showed it as a grounding collapse rather than as the wrong file. See failure
  mode 12 (#13).
