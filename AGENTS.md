# AGENTS.md

Operating instructions for AI coding agents (Hermes, Claude Code, Copilot, etc.)
working in this repository. Read this before changing anything.

## What this project is

A local, open-source rebuild of a multi-agent equity research pipeline:
Python computes every metric deterministically, LLM agents interpret, a Bull and
Bear debate, a verifier grounds every number, and a deterministic rubric scores.
**The rubric verdict is final; no agent can overrule it.** Keep that separation
intact in every change: the LLM layer interprets, the Python layer decides.

## Module map and edit rules

| File | Role | Can edit freely? |
|------|------|------------------|
| `src/data.py` | Evidence fetch: yfinance, SEC EDGAR XBRL. Every value carries a source string. | Yes, but never drop the source strings. |
| `src/metrics.py` | 5 metric pillars. Pure functions of the bundle. | Yes. New metrics must be scored or exempted (see below). |
| `src/rubric.py` | Signal tables, weights, coverage-diluted scoring. The decision layer. | **Owner decision.** Band or weight changes rescore the whole corpus. |
| `src/verify.py` | Matches every agent-emitted number to evidence or derived ratio. | Yes, keep it strict. |
| `src/agents.py` | 9 specialists, Bull, Bear, Judge. Strict JSON, prompt limits, fallback model. | Yes. Read "Prompt bounding" in the README first. |
| `analyze.py` | Orchestration, run artefacts, resume. | Yes. |
| `tests/` | Determinism, sign-direction, grounding invariants. | Extend on every fix. |

## Hard rules

1. **A refused metric is better than a wrong metric.** Missing data must flow
   through as `None` and dilute pillar coverage (see failure mode 9 in the
   README: `share_dilution_pct` refusing a doubled share count). Never
   substitute a plausible number, never zero-fill, never let a corporate action
   score as dilution.
2. **Every computed metric must be scored or explicitly exempted.**
   `test_every_computed_metric_is_scored_or_explicitly_exempt` fails otherwise.
   Adding a metric without a rubric signal is a regression, not an addition.
3. **Signal direction lives in the threshold table only.** The sign-inversion
   bug (failure mode 2 in the README) shipped once. Do not reintroduce a
   direction flag next to an ordered table.
4. **Statement rows match by whole-word tokens.** `_pick_label` exists because
   substring matching once reported Reliance's operating margin as 0.34%
   instead of 11.48%. Do not loosen it.
5. **Fabrication is worse than absence.** A failed agent returns
   `status="unavailable"`, never plausible prose. `finish_reason == "length"`
   is a hard failure: widen the ceiling and retry, never promote a truncated
   response.
6. **README numbers are read back from artefacts, not typed.** When you change
   scoring or metrics, rerun the affected tickers and update
   `runs/*.json` and the README tables from the actual output.
7. **No em dashes (U+2014) anywhere.** Code comments, README, docstrings, PR
   descriptions, commit messages. Use a colon, semicolon, parentheses or a
   new sentence.

## Working conventions

- Python 3.11+. The venv used for this repo on this machine is `/tmp/eq-venv`
  (created for the corpus review; recreate with `pip install -r requirements.txt`).
- Tests run directly, no pytest dependency:
  `python tests/test_metrics.py` and `python tests/test_rubric.py`. Both must
  print their pass count. Run both before every commit.
- The deterministic pipeline needs no credentials: `analyze.py AAPL --no-llm`
  always works. The agent stages need `OLLAMA_API_KEY` in the environment or
  `.env` (never committed).
- Model overrides: `EA_MODEL`, `EA_FALLBACK_MODEL`, `EA_BASE_URL`. Defaults:
  `deepseek-v4.1-flash:cloud`, `glm-5.3:cloud`.
- `runs/` JSON artefacts are the ground truth for what a code change did.
  Before and after a metric or rubric change, diff the artefacts, not just the
  scores.

## Known live defects and owner decisions (do not fix unilaterally)

Tracked from the 2026-09-12 corpus review. These need the owner's decision
before code changes:

1. `share_dilution_pct` definition: whole window vs latest fiscal year vs
   trend. Latest-year changes the score for 62 of 114 corpus tickers
   (48 toward neutral, 14 against). Policy call.
2. Split-aware computation instead of refusal: match a step's ratio to the
   split factor from `price_history["Stock Splits"]`. Constraints from the
   sweep: bound the factor to <= 0.9 or >= 1.5 (near-1 factors are
   distributions and ADR-ratio changes, e.g. SPGI 1.057 is the MBGL
   distribution), and HDB (ADS listing) must yield None: its 154% step stacks
   a bonus with an ADR-ratio change and cannot be repaired by step matching.
3. Basis check: prefer the restated `quarterly_income` series, or refuse the
   annual series when the two disagree for the same period.
4. Window coverage: 2-year price history cannot see splits older than the
   window; `Ticker.splits` or a longer period fixes it.

## Data-source pitfalls (verified on this machine)

- SEC EDGAR rejects `dimitar@localhost` in the User-Agent; use
  `eggressive@example.com` or any RFC-compliant contact. A 403 here is silent.
- NSE India direct returns 403; India coverage comes through yfinance `.NS`.
- Yahoo's `quoteSummary` is crumb-gated (401) for raw HTTP; the library
  handles it, so do not hand-roll requests to Yahoo endpoints.
- yfinance restates the newer annual columns of a split or bonus but not the
  older ones. Any metric that reads a span of annual columns must assume
  mixed bases until the split-aware work lands.

## Review artifacts from the 2026-09-12 sweep

For reference, the corpus review tools live outside the repo (they depend on
a cached 115-ticker pickle corpus): `/tmp/eq-review/sweep_dilution.py` and
`/tmp/eq-review/dilution_followup.md`. Treat their numbers as evidence for the
owner decisions above, not as committed test fixtures.
