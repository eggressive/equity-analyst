# equity-analyst

A local, open-source rebuild of the pipeline described in Shobhit Agarwal's
"I Built 9 AI Agents That Argue Over Every Stock, So You Don't Have To"
(EquityAnalyst.online). Runs locally with free data sources and Ollama Cloud models.

## The pipeline

```
Python calculates  ->  agents interpret  ->  Bull argues  ->  Bear attacks
                   ->  verification checks  ->  deterministic rubric scores
```

The rubric verdict is final. No agent can overrule it. That is the whole point:
the LLM layer generates interpretation, the Python layer generates the decision.

## Modules

| File | Role | LLM involved |
|------|------|--------------|
| `src/data.py` | Evidence fetch: yfinance (US + NSE). Sourced scalar fields carry a source string; SEC EDGAR helpers exist but are not wired in. | No |
| `src/metrics.py` | 5 metric pillars: fundamentals, valuation, technicals, risk, governance. | No |
| `src/rubric.py` | Signal tables, two-horizon weights, coverage-diluted scoring. | No |
| `src/verify.py` | Matches every number an agent emits to an evidence value or a derived ratio. | No |
| `src/agents.py` | 9 specialists, Bull, Bear, Judge. Strict JSON, citation checking, fallback model. | Yes |
| `analyze.py` | Orchestration, run artefacts, resume. | No |
| `tests/test_rubric.py` | Determinism, sign-direction, citation and verifier invariants. | No |
| `tests/test_metrics.py` | Statement-row matching: the silently-wrong-number class of bug. | No |

## Install

```bash
git clone <repo-url> && cd equity-analyst
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+. Only the agent stages need a credential; the deterministic pipeline
runs with none.

```bash
export OLLAMA_API_KEY=...            # any OpenAI-compatible endpoint
cp .env.example .env                 # optional, documents the overrides
```

## Usage

```bash
.venv/bin/python analyze.py AAPL            # full pipeline
.venv/bin/python analyze.py RELIANCE.NS     # NSE
.venv/bin/python analyze.py AAPL --no-llm   # deterministic only, zero API cost
.venv/bin/python tests/test_rubric.py
.venv/bin/python tests/test_metrics.py
```

`--resume <run.json>` reuses the `ok` stages of an earlier run (`--out` writes the file
to reuse). No resume fixture ships with this repository, so point it at your own.

Env overrides: `EA_MODEL` (default `deepseek-v4.1-flash:cloud`),
`EA_FALLBACK_MODEL` (default `glm-5.3:cloud`), `EA_BASE_URL`.

No Hermes dependency: the project imports no Hermes modules and reads no Hermes
config. Point `EA_BASE_URL` at any OpenAI-compatible server, or use `--no-llm` to
skip the LLM entirely.


## Scoring

Signal scale is -2..+2. Pillar score is the mean of its *available* signals;
missing signals are excluded rather than scored zero, and the pillar's coverage
fraction dilutes its weight. If total confidence falls below 0.45 the result is
`INSUFFICIENT_DATA` instead of a guess.

Not every signal can reach the ends of that scale. Several tables top out at +1
(revenue growth, cash conversion, both trend signals) and four are confined to -1..+1
(RSI, distance from the 52-week high, institutional and insider holding); `max_drawdown`
cannot reach -2. The practical effect is that the technicals pillar ranges about -1.5..+1.0
while carrying the largest SHORT_TERM weight, so a short-term verdict is harder to push to
`STRONG_BULLISH`/`STRONG_BEARISH` than the -2..+2 label suggests.

Two horizons with different weights:

- `SHORT_TERM` (2-8 weeks): technicals 0.35, risk 0.30, valuation 0.15, fundamentals 0.10, governance 0.10
- `LONG_TERM` (3-5 years): fundamentals 0.32, valuation 0.26, governance 0.20, risk 0.15, technicals 0.07

## Verified runs (2026-09-12, current code)

| Symbol | SHORT_TERM | LONG_TERM | Agents | Grounding | Unmatched | Time |
|---|---|---|---|---|---|---|
| AAPL | NEUTRAL +0.294 | BULLISH +0.458 | 9/9 | 0.968 | 0 | 51s |
| RELIANCE.NS | BULLISH +0.500 | BULLISH +0.734 | 9/9 | 0.977 | 0 | 140s |
| TCS.NS | NEUTRAL +0.246 | BULLISH +0.796 | 9/9 | 0.966 | 0 | 118s |

Artefacts: `runs/AAPL_final.json`, `runs/RELIANCE_full.json`, `runs/TCS_full.json`.
Verdicts, scores, grounding rates, agent counts and run times are read back from those
files. Two caveats. The AAPL run is a *resumed* run: it reused nine specialist outputs and
the bull from an earlier file, so its 51s covers the bear, verification and judge only,
and that earlier file is not part of this repository. And the construction-time token
counts quoted in the failure-mode list below are not all reproducible from the tracked
artefacts; where that is the case it now says so.

The Bear earned its place on AAPL. Its six attacks in `runs/AAPL_final.json` go after the
bull's load-bearing claims: that trailing margins prove durable pricing power (the bundle
carries no peer benchmark or segment detail, so they cannot); that $98.77B of FCF funds
buybacks "without external capital" (a ~2.0% FCF yield on a $4.85T market cap means
buybacks retire shares at 38.15x earnings); that one annual OCF/NI of 1.0 is evidence of
clean earnings quality (one ratio is not a trend); that 31.18% ROA proves a productive asset
base (a trailing ratio with D/E 1.34 and cash at 10.0% of assets); and that a "confirmed
uptrend" is a low-risk entry (RSI14 70.62 is overbought on volume 21.86% below its 20-day
average). The judge recorded the same six as `bull_case_broken`.

The debate cannot move the verdict. The rubric is computed from the metric bundle before
any agent runs, so the debate changes the narrative only. That is a property of the code,
not of these runs: no artefact stores a before/after verdict pair.

**Unmatched numbers are zero across all three tickers**: every figure any agent wrote
resolved either to an evidence value or to arithmetic over evidence values. Read that
carefully, because the check is weaker than it sounds. Matching is value-level, not
claim-level, and the derived index is generous: 52 AAPL evidence values generate ~4,000
derived values, and that index accepts roughly 95% of random numbers in 0.05..100. So
`unverified = 0` means "nothing obviously invented", not "every claim is grounded"; 12.9% of
AAPL's 342 numbers matched only through the derived index. Two further accounting notes: the
verifier skips bare 4-digit years after fixing the denominator, so the AAPL/RELIANCE/TCS
`claims_found` of 342/258/175 includes 11/6/6 numbers that are neither verified nor
unverified; and `runs/RELIANCE_full.json` records one citation violation, the `flows` agent
citing `pillar_coverage.governance`.



## What this is missing versus EquityAnalyst.online

Honest gaps, in rough order of impact:

1. **Peer and historical multiples.** The system can compute AAPL's PE but not
   AAPL's PE *versus its own 5-year range or versus peers*. This is the single
   biggest analytical hole: both the Bull and the Bear flagged it unprompted.
2. **Consensus estimates.** `forwardPE` comes from Yahoo; there is no real analyst
   consensus series, so forward-looking claims rest on one provider number.
3. **Earnings transcript and filing text.** Founder Desk and Earnings Call
   Analyzer need document ingestion (10-K/Q + transcripts) and a retrieval layer.
   Not started. The SEC EDGAR helpers in `src/data.py` are not called by the pipeline
   at all yet, so no number in a run comes from a filing.
4. **News sentiment actually scored.** Yahoo's headline fields came back null for
   AAPL, so the sentiment agent correctly reported "unscoreable" rather than
   inventing a tone. A real source (news API or RSS + extraction) is needed.
5. **Flows and macro data.** Institutional holdings exist via yfinance but
   changes over time do not. No rates, FX, or index series are wired in, so the
   macro agent can only reason from single-name proxies.
6. **Technical depth.** Trend, RSI, ATR, volatility and drawdown exist. No
   support/resistance detection, no volume profile, no relative strength.
7. **Unscored metrics.** `gross_margin_pct`, `atr14_pct`, `volume_vs_20d_avg`,
   `short_ratio`, `audit_risk`, `overall_risk` and `compensation_risk` are computed
   and cited by agents but have no rubric signal. They are registered as exempt
   with reasons in `tests/test_rubric.py`, so adding one requires a deliberate edit.
8. **No UI.** CLI and JSON artefacts only. No Analyze screen, Compare, Journal,
   Watchlist, or portfolio tracking.
9. **No persistence layer.** Runs are files. No queryable history, no "what
   changed since the last run".

## Known failure modes, already handled

Nine bugs shipped and were fixed during construction. Bugs 1-5 and 9 are Python errors in
the metric and verification layers; 6 and 7 are LLM-protocol errors; 8 is an HTTP 403 from
SEC EDGAR. `tests/` guards bugs 1-5 and 9. Bugs 6, 7 and 8 have no test and are recorded
here only, which is a real gap in the suite rather than a claim of coverage.

1. **Wrong statement row.** Substring matching let `Other Non Operating Income
   Expenses` satisfy the needle `"Operating Income"`, so Reliance's operating margin
   was reported as **0.34% instead of 11.48%**, and the Bear built a confident thesis
   on it. `_pick_label` now requires the needle tokens to appear as a contiguous run,
   refuses any candidate carrying a meaning-changing token (`non`, `other`,
   `excluding`, ...), drops the raw-substring fallback, and prefers a leading match
   with the fewest extra tokens. Guarded by `tests/test_metrics.py`, which now also
   covers the second instance of the same bug, found later: with no true
   `Operating Income` row, BRK-B reported **-1.23%** from
   `Net Non Operating Interest Income Expense` and HDFCBANK.NS **-1.91%** from
   `Other Non Operating Income Expenses`, both scoring the `operating_margin` signal
   at -2. BRK-B now falls through to its real `EBIT` row (21.32%); HDFCBANK.NS has no
   such row, so the metric stays `None` and the coverage fraction dilutes its weight
   instead of a fabricated number scoring the signal.
2. **Sign inversion.** An `higher_is_bullish` flag applied to already-ordered
   threshold tables flipped every signal, scoring a low-debt test company as highly
   leveraged. Direction now lives in the table only. Guarded by
   `test_low_leverage_scores_positive_and_high_scores_negative`.
3. **Unscored metric.** `operating_margin_pct` was computed and cited by agents but
   absent from the signal table, so the 0.34% → 11.48% correction changed the
   narrative and left the verdict untouched. Now a signal, and
   `test_every_computed_metric_is_scored_or_explicitly_exempt` fails unless every
   computed metric is mapped to a signal or exempted with a written reason.
4. **Verifier phantom.** `"Beta of 1.08 means"` parsed as `1.08 million`, because the
   optional unit group ate the `m` of "means". Fixed with a trailing negative
   lookahead. Guarded by `test_verification_ignores_unit_letter_inside_a_word`.
5. **Verifier blind spot.** Sourced `extras` values (institutional holder amounts)
   were flagged as ungrounded. `_flatten` now walks nested extras. Guarded by
   `test_verification_credits_sourced_extras`.
6. **Token starvation.** Long JSON payloads pushed the Bear past its output ceiling
   on both models, so it returned truncated JSON and reported `unavailable`. Fixed
   by slimming the debate payload and treating `finish_reason == "length"` as a hard
   failure that widens the ceiling and retries. A truncated response is never
   promoted to an answer.
7. **Output proportional to input.** Even after (6), the Bear still truncated: the
   Bull emitted up to 12 verbose arguments, so the Bear generated unbounded rebuttals.
   The saved truncation record is `max_tokens=4860` with ~7,428 tokens used on the
   primary model and ~7,326 on the fallback. Widening the cap repeatedly does not
   converge, because the output scales with the input. Fixed by hard word and item
   limits in the bull/bear prompts plus trimming the bull's arguments to 6 in the
   payload that the Bear receives. The Bear dropped from `unavailable` to 6 attacks
   at 7,076 tokens.
8. **SEC 403.** Measured 2026-09-12. `data.sec.gov` and `www.sec.gov` answer 403 with a
   page titled "Your
   Request Originates from an Undeclared Automated Tool" when the User-Agent is a
   library or bot default: `curl/8.5.0`, `Wget/1.21.4`, `Python-urllib/3.14` and
   `python-requests/2.32.3` are blocked on `companyfacts`, `submissions`,
   `company_tickers.json` and `xbrl/frames`, and eight rapid requests with a
   descriptive string stay at 200. A descriptive product token passes and the contact
   is not validated: `EquityAnalyst-Research/0.1` with no contact, with
   `dimitar@localhost`, or with `not-an-email` all return 200. Keep a contact anyway,
   because SEC's policy asks for one and enforcement is theirs to change. The SEC
   helpers in `src/data.py` are still called by nothing, so no run depends on this.

9. **Corporate action read as dilution.** `share_dilution_pct` is now the change implied by
   the least-squares line through log(average shares) across the columns yfinance returns.
   Three mechanisms replaced the first guard, which refused the metric whenever a doubling
   or halving coincided with a split inside the statement window.

   **The step is repaired, not refused.** yfinance can restate the annual columns of a bonus
   issue for the newer periods and leave the older ones, so one series mixes two bases:
   HDFCBANK.NS carries 7.107bn shares for FY2024 (pre-bonus) next to 15.319bn for FY2025
   (post-bonus), a step of 2.155x against its recorded 1:1 bonus of 2025-08-26, and first
   against last read the window as **+175.75%**, scoring the `dilution` signal at -2. The step
   is that bonus plus the year's real issuance (2.155 = 2.0 x 1.078), so the older columns are
   rebased and the metric reports what is left, **+36.56%**: the merger, which keeps its -2.
   The factor comes from the full split history (`Ticker.splits`), not the two-year price
   series, because 9 of the 17 in-window splits in this sample are older than that window
   (NVDA 10:1 on 2024-06-10, NVO 2:1 on 2023-09-20, GE 1.281 and 1.253, MMM 1.196). Only
   ratios at or below 0.9 or at or above 1.5 count as factors: near-1 ratios are distributions
   and ADR-ratio changes (SPGI 1.057, HON 0.9535, UL 0.888) and cannot rebase a share count.

   **A series that still mixes bases falls back to the restated quarterly columns.** An
   implausible step in each direction means two bases rather than two corporate actions: HDB
   (ADS listing,
   3.709bn -> 1.862bn -> 4.738bn) and TRV (a 23.11bn column where the years either side are
   0.23bn). TRV reads -7.04% from its quarterly columns, a buyback that keeps its +2. HDB has
   only two quarterly columns, which is a comparison rather than a trend, so it is refused.
   The same fallback is the basis check: a disagreement between the annual and quarterly
   columns for one fiscal period puts the restated quarterly series in charge, which moves
   COF (14.3% apart), INTC (6.71%) and BA (5.16%) onto it.

   **Issuance keeps its score.** A count that doubles with no split behind it is an all-stock
   acquisition or sustained equity financing and keeps its -2, because refusing it would
   delete a real dilution penalty and flatter the company. The widest real cases in the sample
   are Realty Income at +48.5% (repeated raises) and AIG at -27.6% (sustained buybacks).

   Across the 115-ticker sample the change moves six dilution signals (ALL -1 to 0, PDD -1 to
   -2, TSLA -1 to +1, XOM -1 to -2, HDFCBANK.NS refused to -2, HDB -2 to refused) and one
   verdict: HDB LONG_TERM NEUTRAL 0.313 to BULLISH 0.497, because a wrong -2 leaves the
   pillar. Governance coverage moves with the refusals: HDB 1.000 to 0.667, HDFCBANK.NS 0.667
   to 1.000. **The tracked runs predate this change, and none of their machine-readable content
   does.** `runs/*.json` record the 2026-09-12 paid runs, and no signal or score in them moves,
   because no band moves: AAPL keeps +2 while the value quoted in its prose drifts -8.09% to
   -8.03% (11 mentions), and TCS keeps +1 while its prose drifts -1.12% to -1.24% (7 mentions).
   Repairing that prose means paying for new runs. The deterministic half is free: on
   2026-09-13 `python analyze.py <SYM> --no-llm` reproduced all three published rows from live
   data into a timestamped artefact of its own, AAPL NEUTRAL 0.294 / BULLISH 0.458,
   RELIANCE.NS BULLISH 0.5 / 0.734 and TCS.NS NEUTRAL 0.246 / BULLISH 0.796.
   Guarded by `tests/test_metrics.py`, which covers the bonus repair, the quarterly fallback,
   the refusal when neither series is usable, the trend reading against the endpoint reading,
   one factor repairing one step, a reverse split with issuance, and a rubric check that a
   refused value is unavailable rather than scored.

**Fabrication is treated as worse than absence.** A failed agent returns
`status="unavailable"` with an error string, never plausible-looking prose: a truncated
response is detected from `finish_reason == "length"`, retried, and never promoted to an
answer. Earlier drafts of this paragraph cited a run where 5 of 9 agents failed and a TCS
run that reported `agents 8/9`. No artefact for either is in this repository, so both have
been removed: the guarantee rests on the code path, not on a tracked run.

## Prompt bounding is part of the protocol

Failure mode 7 generalises: on a chat-completions API you control only `max_tokens`,
and these models do not stop at a requested JSON size. Every agent prompt therefore
carries explicit hard limits (max items, max words per field), and the bull's arguments
are trimmed to 6 before the bear sees them. Without both, the agent that fails is
always the adversarial one, which is the worst possible component to lose silently.



## Data sources

- **yfinance** 1.7.0 — prices, OHLCV, statements, split histories, quote metadata. US and NSE
  (`.NS`). No key. Yahoo's `quoteSummary` endpoint is crumb-gated (401) for raw
  HTTP; the library handles it.
- **SEC EDGAR XBRL** — `data.sec.gov/api/xbrl/companyfacts`. Free, requires a descriptive
  User-Agent. `src/data.py` implements `sec_companyfacts` and `resolve_cik`, but nothing
  calls them yet, so every number in a run currently comes from yfinance.
- **NSE India direct** returns 403 from the machine this was built on; India coverage comes
  through yfinance `.NS` symbols, which work.

## Scope

Research tooling, personal use. Two independent horizon verdicts are structured
interpretations of evidence, not instructions to trade. DCF-style multiples are
estimates. No personalized financial advice.
