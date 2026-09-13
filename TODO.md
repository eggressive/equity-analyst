# TODO

Open findings from the 2026-09-13 codebase validation. Resolved portions are noted below.
Completed changes belong in [CHANGELOG.md](CHANGELOG.md).

## Validation baseline

- **Tests:** `tests/test_metrics.py` passed 36/36; `tests/test_rubric.py` passed 44/44.
- **Live check:** AAPL completed with `--no-llm`. Paid LLM calls were not tested.
- **Evidence:** Edge cases below were reproduced with synthetic inputs where stated; corpus impact is unknown.
- **Implementation rules:** Follow [AGENTS.md](AGENTS.md), including its owner decisions, defect-registration rule and artifact refresh requirements. Add regression tests for each fix.

## High priority

- [ ] **T01: Refuse misleading ratios with negative denominators.** In `src/metrics.py`, negative equity and net income can turn distress into bullish signals. A fixture with debt 100, equity -10, net income -20 and operating cash flow -40 produced leverage +2, ROE +2 and cash conversion +1. Decide the valid denominator policy before implementation. Test that invalid ratios become unavailable and dilute coverage without changing rubric thresholds.
- [ ] **T02: Bind resumed analysis to its evidence.** `analyze.py` fetches fresh evidence before reusing saved prose without checking evidence identity. Cross-ticker reuse was fixed in commit `1f5e1f9` (PR #13); missing symbols still produce a warning. Missing specialists remain unavailable instead of being retried. Persist the evidence bundle and fingerprint, and define whether resume restores the original snapshot or reruns stages against fresh evidence. Test changed evidence, missing identity and partial failures; retain the existing cross-ticker regression test.
- [ ] **T03: Prevent repeated split repair of one step.** `_repair_share_steps` in `src/metrics.py` can assign multiple recorded factors to the same unchanged step. A synthetic annual series `[100, 200, 200]` with two recorded 2:1 factors became `[400, 200, 200]`. Track consumed steps or explicitly reconcile compound actions. Decide how ambiguous multiple-factor matches should fall back or be refused. Test separate splits and multiple actions within one statement interval.

## Medium priority

- [ ] **T04: Validate agent response schemas.** `call_agent` in `src/agents.py` accepts an empty JSON object as `status="ok"` with no violations, reproduced with a mocked response. Validate required fields, types and score ranges for each role. Test empty objects, malformed fields and out-of-range scores through retry and unavailable-result paths.
- [ ] **T05: Handle unusable closing prices.** `technicals` in `src/metrics.py` raises `IndexError` when a nonempty history has only null closes. Return unavailable technical metrics after filtering leaves no usable prices; guard `risk` against missing close columns too. Test that the full deterministic pipeline degrades coverage instead of crashing.
- [ ] **T06: Handle zero-loss RSI windows.** `technicals` in `src/metrics.py` converts infinite relative strength to NaN. Thirty strictly rising closes reproduced `rsi14=None`. Handle zero-loss windows explicitly, decide the flat-window convention, and test rising, falling, flat and insufficient-history inputs.

## Further fixes and optimizations

- [ ] **T07: Correct or rename ATR.** `technicals` in `src/metrics.py` averages high minus low and excludes gaps from the previous close. Define true range, smoothing and minimum history explicitly. Add an overnight-gap fixture with a hand-calculated result.
- [ ] **T08: Define downside deviation and its window.** `risk` in `src/metrics.py` takes standard deviation of negative returns and selects the last 252 observations after filtering. Decide the target return, denominator and observation window. Test against a hand-calculated series containing positive, negative and zero returns.
- [ ] **T09: Verify judge numbers.** `analyze.py` invokes the judge after numeric verification, so judge prose is excluded. Verify it against evidence and the deterministic results it receives, with per-agent attribution. Keep verification informational and assert that it cannot change rubric scores or verdicts.
- [ ] **T10: Bound payloads before serialization.** `call_agent` in `src/agents.py` slices serialized JSON at 14,000 characters, which can cut a field or remove required evidence. Reduce optional content structurally, then serialize. Test oversized payloads for valid JSON and preservation of required evidence and instructions.
- [ ] **T11: Reuse technical calculations.** `all_pillars` in `src/metrics.py` computes technicals directly and again through `risk`. Pass the computed result through while preserving standalone `risk` calls. Check output equivalence and measure the benefit before claiming a speedup.
