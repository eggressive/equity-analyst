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
| `src/verify.py` | Matches every agent-emitted number to evidence or to a derived value whose metrics are named beside it. | Yes, keep it strict. |
| `src/agents.py` | 9 specialists, Bull, Bear, Judge. Strict JSON, enforced output limits, fallback model. | Yes. Read "Prompt bounding" in the README first. |
| `analyze.py` | Orchestration, run artefacts, resume. | Yes. |
| `tests/` | Determinism, sign-direction, grounding invariants, and the document rules in `tests/test_docs.py`. | Extend on every fix. |

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
   `python analyze.py <SYM> --no-llm` does the deterministic half for free and writes
   its own timestamped artefact: use it to re-derive the published scores. The tracked
   `runs/*.json` are dated records of a paid run, so their prose carries the values of the
   code that produced it. **Refresh the affected tickers rather than splice numbers into an
   old run**: one paid run per ticker costs about 50k tokens and two to three minutes, and a
   spliced record claims the agents said something they did not.
7. **No em dashes (U+2014) anywhere.** Code comments, README, docstrings, PR
   descriptions, commit messages. Use a colon, semicolon, parentheses or a
   new sentence.
8. **Output limits live in code, not in the prompt.** `check_limits` enforces the caps
   the prompts state, and the `evidence` path beside every Bull argument and every Bear
   attack is checked like any other citation. A new agent needs its limit table, or its
   caps are prose again. This covers agent prose in `runs/*.json` too: the model is asked not to
   emit the character, and `strip_em_dashes` in `src/agents.py` rewrites it to a comma on
   the way in, so a run cannot ship one.

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
- **GitHub work goes through the GitHub MCP tools, not the REST API.** When
  reading or reviewing PRs, issues, branches or checks, prefer the GitHub MCP
  server's tools over `gh` shell calls or hand-rolled `curl` against
  `api.github.com`: it returns structured output instead of text to parse. Raw
  `curl` requests to the GitHub API are a last resort, only when an MCP tool for
  the operation does not exist. Plain `git` commands (clone, commit, push, rebase)
  are unaffected: this rule covers reading and commenting on GitHub state,
  not local repository work.

  Three traps, all hit in practice:

  - **The connection is per session.** A server added from the shell is visible to
    new sessions and after `/reload`, not to the running one. Until the reload,
    every call raises `KeyError: "MCP server '<name>' is not declared in user
    settings"`. Reload rather than falling back to `gh` or `curl` silently.
  - **Discover before calling.** Tool names and argument schemas come from
    `mcp.list_tools("<server>")`: each entry carries `name`, `description` and the
    `inputSchema` that is the real signature. The kernel's help covers only
    `list_tools`, `call_tool`, `reload` and `close`, never a server's tools.
  - **Parse the result.** `mcp.call_tool` returns a JSON string for this server,
    not a dict.

## Writing style for every document you touch

Applies to `AGENTS.md`, `README.md`, docstrings, PR descriptions and code
comments alike.

- **Formatting:** bold lead-ins on bullets, backticks for commands and paths,
  fenced blocks with a language tag.
- **Voice:** terse and opinionated. Name unknowns as unknowns rather than
  writing around them, and state caveats and traps explicitly.
- **One reason per claim, at most.** The failure mode is a justifying clause
  on every sentence: enjoyable once, exhausting at volume, and it buries
  the decisions.
- **Rationale only where contested or counter-intuitive; elsewhere state it
  and move on.** Prefer a table to a paragraph.
- **Relative links between documents, including to anchors:** `[Part 5](GitHub/copilot-pilot.md#part-5--decide)`.
- **Don't duplicate content across documents.** Link to the one that owns
  the topic.

## Changelog rules (CHANGELOG.md)

`CHANGELOG.md` owns what changed between releases. `git log` owns every change at commit
granularity, `README.md` owns current behaviour and its measurements, `TODO.md` owns open
work. Merge is a squash, so one merged PR is one entry.

- **Write the entry in the PR that makes the change.** The evidence that justifies it is in
  that diff, and the entry lands with it.
- **One to three lines**, in user-visible terms, with the PR number. Link the README section
  that owns a measurement instead of restating a number: two copies of a number drift.
- **Mark the three cases a reader of an old artefact has to know.**
  - **Rescore:** verdicts move, so committed `runs/*.json` are stale. Name the runs re-run.
  - **Schema:** `runs/*.json` gained or lost a key, so consumers of the artefact break.
  - **Cost:** the change implies paid LLM calls, as the refresh in PR #11 did.
- **Never** put measurements, root causes or test counts in an entry. The README failure
  modes and the PR body own those.
- **Release:** promote `Unreleased` to `## [0.1.0] - YYYY-MM-DD`, tag it, and use the section
  body as the GitHub Release body. Keep `## [Unreleased]` present at all times: an artefact
  records the model and the timestamp but not the code, so the tag is the only link from a
  run to the revision that produced it.
- **Checked mechanically** by `tests/test_docs.py`: the Unreleased section, dated version
  headings, resolvable links and anchors, named `runs/` files, recorded test counts, and no
  em dash.

## Verification scope (src/verify.py)

- **Names, not just values.** A number matches an evidence value, or a derived value whose
  two metrics are named beside it. Extend `METRIC_ALIASES` when agents start citing a metric
  by a name the table does not cover, or honest prose gets flagged.
- **`unverified == 0` is not proof of grounding.** Value-level matching still accepts 27% of
  random numbers in 0.05..100 on an AAPL bundle (measured 2026-09-13). The paragraph-level
  guard is `test_a_fully_invented_paragraph_is_reviewed`.
- **The count, not the list.** `analyze._verify_extras` takes `news_count` from the bundle.
  Recomputing it from the headline list writes zero, because the bundle no longer carries
  the list.
- **REVIEW is informational.** It appends a warning and `verification.unverified_by_agent`.
  It must never reach a score: the rubric is computed before any agent runs.

## Owner decisions on the share-count metric (2026-09-13)

All four items from the 2026-09-12 review are decided and implemented.
`README.md` failure mode 9 owns the measurements; the code is in `src/metrics.py`.

| Item | Decision |
|------|----------|
| `share_dilution_pct` definition | **Trend**, not the window endpoints: the change implied by the least-squares line through log(average shares). |
| Split-aware computation | **Compute**. A step a recorded split factor explains is repaired instead of refusing the metric. Factors are bound to <= 0.9 or >= 1.5, which keeps distributions and ADR-ratio changes out. HDB still yields None, now because no factor matches its steps. |
| Basis check | The restated `quarterly_income` series is measured when the annual and quarterly columns disagree for one fiscal period, or when the annual steps stay implausible after repair. |
| Split-history window | The full history from `Ticker.splits`, one extra call, not the two-year price series. |

## Known live defects (do not fix unilaterally)

The register is [`TODO.md`](TODO.md): the open findings from the 2026-09-13 validation,
with the reproduction evidence and the decision each one needs. Read it before changing
behaviour it describes, and add an entry rather than leaving a defect untracked. Decided
items move into a decisions section here, like the share-count metric above.

## Data-source pitfalls (verified on this machine)

- SEC EDGAR blocks undeclared automated tools: `curl/8.18.0`, `Wget2/2.2.1`,
  `Python-urllib/3.14` and `python-requests/2.34.3` return 403 on `data.sec.gov` and
  `www.sec.gov`, while a descriptive User-Agent stays at 200. The two hosts then
  differ: on `data.sec.gov` (the only SEC host this repo calls, in `src/data.py`)
  the contact is not validated, `dimitar@localhost` included; on `www.sec.gov` a
  plausible email is required and `localhost` returns 403. Keep one anyway, for
  SEC fair-access policy rather than for the 403. Re-check:

  ```bash
  curl -H "User-Agent: EquityAnalyst-Research/0.1 (personal research; eggressive@example.com)" \
    https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json
  ```
- NSE India direct returns 403; India coverage comes through yfinance `.NS`.
- Yahoo's `quoteSummary` is crumb-gated (401) for raw HTTP; the library
  handles it, so do not hand-roll requests to Yahoo endpoints.
- yfinance restates the newer annual columns of a split or bonus but not the
  older ones, so any metric that reads a span of annual columns must assume
  mixed bases. `share_dilution_pct` repairs a step a recorded factor explains and
  falls back to the restated quarterly series otherwise; no other metric does yet.
- Ollama cloud models on this workstation go through the local daemon, not
  `https://ollama.com/v1`: `OLLAMA_API_KEY` here is the placeholder `ollama-local`, so a
  direct call returns 401 and all nine agents report `unavailable`, while `127.0.0.1:11434`
  holds the account credential and serves the same cloud models. Run with the placeholder
  key and `EA_BASE_URL=http://localhost:11434/v1`. Re-check:

  ```bash
  curl -s http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4.1-flash:cloud","messages":[{"role":"user","content":"ok"}],"max_tokens":4}'
  ```

## Review artifacts from the 2026-09-12 sweep

For reference, the corpus review tools live outside the repo (they depend on
a cached 115-ticker pickle corpus): `/tmp/eq-review/sweep_dilution.py` and
`/tmp/eq-review/dilution_followup.md`. Treat their numbers as evidence for the
owner decisions above, not as committed test fixtures.
