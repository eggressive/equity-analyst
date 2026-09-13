"""The 9 specialist agents, the Bull, the Bear, and the synthesizer.

Contract enforced here:
  - An agent NEVER fetches data and NEVER invents a number. It receives a
    deterministic bundle and may only interpret values inside it.
  - Every agent returns strict JSON with a `cited_metrics` list naming the keys
    it used. Unknown keys are dropped and recorded as a violation.
  - A failed agent returns status="unavailable". It never returns prose that
    looks like analysis. Missing beats fabricated.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from openai import OpenAI

DEFAULT_MODEL = os.environ.get("EA_MODEL", "deepseek-v4.1-flash:cloud")
FALLBACK_MODEL = os.environ.get("EA_FALLBACK_MODEL", "glm-5.3:cloud")
BASE_URL = os.environ.get("EA_BASE_URL", "https://ollama.com/v1")

SPECIALISTS = {
    "fundamentals": "revenue growth, margins, cash conversion, balance sheet strength, capital allocation",
    "valuation": "historical multiples, peer-relative multiples, and what the current price implies",
    "technicals": "trend, momentum, volume behaviour, volatility, and risk levels",
    "sentiment": "recent news flow and whether the information may already be priced in",
    "events": "earnings, guidance, corporate actions, and calendar risk",
    "flows": "institutional positioning and changes in holdings",
    "macro": "index and rates backdrop, sector and currency conditions",
    "governance": "ownership structure, insider activity, dilution, audit and compensation risk",
    "risk": "volatility, drawdown, beta, concentration, and position sizing implications",
}

AGENT_SYSTEM = """You are the {name} analyst agent on an equity research desk.
Your remit: {remit}.

Hard rules:
1. You may ONLY use numbers that appear in the JSON evidence bundle given to you.
2. Every number you state must also be listed in "cited_metrics" using the exact key path from the bundle.
3. If the evidence for a claim is missing or thin, say so in "evidence_gaps" instead of guessing.
4. Obey the provenance contract in "data_quality": fields listed in
   "no_time_series_for" are single current observations, so never describe them as
   rising or falling over a period. Fields in "no_source_for" simply do not exist in
   this bundle; do not reason as if they do.
5. Do not give a buy or sell verdict. Score only your own remit on a -2..+2 scale.
6. Be concise and specific. No filler, no disclaimers.
7. Never use an em dash (U+2014). Use a comma, colon, semicolon, parentheses or a new
   sentence; the pipeline rewrites the dash to a comma if one slips through.

Output limits are hard requirements. "summary" at most 60 words, at most 5 key_points
at most 25 words each, at most 5 cited_metrics, at most 4 evidence_gaps. Do not restate
the evidence bundle. Do not pad.

Return ONLY valid JSON:
{{
  "agent": "{name}",
  "score": <float -2..2>,
  "confidence": <float 0..1>,
  "summary": "<max 60 words>",
  "key_points": ["<max 25 words>"],
  "cited_metrics": ["<key path>"],
  "evidence_gaps": ["<what is missing>"]
}}"""

BULL_SYSTEM = """You are the BULL analyst. You must build the strongest honest bull case for this stock.
Use ONLY the evidence bundle. The bear agent will attack your case, so do not overstate.
Never use an em dash (U+2014); use a comma, colon, semicolon, parentheses or a new sentence.

Output limits are hard requirements. Give at most 5 arguments. Each "claim" must be at
most 40 words. Keep "thesis" to at most 80 words and list at most 4 break conditions.
Do not restate the evidence bundle. Do not pad.

Return ONLY valid JSON:
{
  "role": "bull",
  "thesis": "<max 80 words>",
  "arguments": [{"claim": "<max 40 words>", "evidence": "<exact key path from bundle>"}],
  "key_metrics": ["<key path>"],
  "what_would_break_this": ["<condition>"]
}"""

BEAR_SYSTEM = """You are the BEAR analyst. Your job is to break the bull case, not to be balanced.
Attack the most load-bearing bull arguments, using ONLY the evidence bundle. If the bull
cited a metric, check it. If the bull overstated, say which claim fails and why.
Never use an em dash (U+2014); use a comma, colon, semicolon, parentheses or a new sentence.

Output limits are hard requirements. You have a maximum of 6 attacks. Each attack's
"counter" must be at most 45 words. Keep "rebuttal" to at most 90 words and list at
most 4 unresolved questions. Do not restate the evidence bundle. Do not pad.

Return ONLY valid JSON:
{
  "role": "bear",
  "rebuttal": "<max 90 words>",
  "attacks": [{"bull_claim": "<short>", "counter": "<max 45 words>", "evidence": "<exact key path from bundle>"}],
  "strongest_bear_metric": {"key": "<key path>", "value": <number>},
  "unresolved": ["<question>"]
}"""

JUDGE_SYSTEM = """You are the synthesis judge. You receive the 9 specialist outputs, the bull case,
the bear's rebuttal, a mechanical verification report, and the deterministic rubric result.
The rubric already decided the verdict. You do not change it.
Never use an em dash (U+2014); use a comma, colon, semicolon, parentheses or a new sentence.
Your job is to explain the verdict in one paragraph and to list the top open questions.

Return ONLY valid JSON:
{
  "explanation": "<paragraph tying the verdict to the evidence>",
  "bull_case_held": ["<bull argument that survived>"],
  "bull_case_broken": ["<bull argument the bear dismantled>"],
  "open_questions": ["<question>"],
  "biggest_uncertainty": "<one sentence>"
}"""


@dataclass
class AgentResult:
    name: str
    status: str
    data: dict
    raw: str = ""
    error: str = ""
    model: str = ""
    tokens: int = 0
    violations: list[str] = None

    def as_dict(self) -> dict:
        return {
            "agent": self.name,
            "status": self.status,
            "data": self.data,
            "model": self.model,
            "tokens": self.tokens,
            "error": self.error,
            "violations": self.violations or [],
        }


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY") or "local"
        _client = OpenAI(api_key=key, base_url=BASE_URL)
    return _client


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# AGENTS.md rule 7 bans U+2014 anywhere, and agent prose is written by a model that
# emits it whatever the prompt says. The dash is rewritten on the way in rather than
# left to the prompt, because a rule the pipeline depends on has to hold in Python.
# A comma reads correctly in both forms the models emit: the spaced one ("growth is
# slow, margins are not") and the unspaced one used as a parenthetical pair
# ("valuation, 38.15x book, can be justified").
EM_DASH_RE = re.compile(r"\s*\u2014\s*")


def strip_em_dashes(value):
    """Replace every em dash with a comma in model-produced text, recursively."""
    if isinstance(value, str):
        return EM_DASH_RE.sub(", ", value)
    if isinstance(value, dict):
        return {k: strip_em_dashes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_em_dashes(v) for v in value]
    return value


# Output limits are stated as hard requirements in every prompt, and a prompt is not
# an enforcement mechanism: the models overran them in the 2026-09-12 runs (a 12
# argument bull case against a stated maximum of 5, 8 of 9 summaries over 60 words,
# a 143 word thesis against 80). Same reasoning as rule 7: a rule the pipeline depends
# on has to hold in Python. Each entry is (path, kind, cap), where "[]." addresses a
# field inside a list of objects.
SPECIALIST_LIMITS = (
    ("summary", "words", 60),
    ("key_points", "items", 5),
    ("key_points[]", "words", 25),
    ("cited_metrics", "items", 5),
    ("evidence_gaps", "items", 4),
)
BULL_LIMITS = (
    ("thesis", "words", 80),
    ("arguments", "items", 5),
    ("arguments[].claim", "words", 40),
    ("what_would_break_this", "items", 4),
)
BEAR_LIMITS = (
    ("rebuttal", "words", 90),
    ("attacks", "items", 6),
    ("attacks[].counter", "words", 45),
    ("unresolved", "items", 4),
)
# The judge's prompt states no numeric caps: it explains a verdict and lists open
# questions, so there is nothing to count.
LIMITS_BY_AGENT = {"bull": BULL_LIMITS, "bear": BEAR_LIMITS, "judge": ()}

# The bear is shown the bull's case, trimmed to the protocol caps so its input stays
# bounded. Shared by the run and the resume path, which had separate copies of this.
BULL_ARGUMENT_CAP = 5
BREAK_CONDITION_CAP = 4


def _words(value) -> int:
    return len(value.split()) if isinstance(value, str) else 0


def check_limits(data: dict, limits, trim: bool = True) -> tuple[list[str], list[str]]:
    """Enforce the stated output limits. Returns (violations, retryable).

    Item overruns are trimmed in place and reported: the protocol caps the count, and
    a downstream agent is shown a trimmed view anyway, so the stored output should
    match what the rest of the pipeline sees. Word overruns are reported as retryable
    and never trimmed, because cutting prose mid-sentence loses meaning silently.
    """
    violations: list[str] = []
    retryable: list[str] = []
    for path, kind, cap in limits:
        if "[].claim" in path or "[].counter" in path:
            head, field = path.split("[].", 1)
            for i, item in enumerate(data.get(head) or []):
                if isinstance(item, dict) and _words(item.get(field)) > cap:
                    msg = (f"over limit: {head}[{i}].{field} is "
                           f"{_words(item.get(field))} words (max {cap})")
                    violations.append(msg)
                    retryable.append(msg)
            continue
        if path.endswith("[]"):
            head = path[:-2]
            for i, item in enumerate(data.get(head) or []):
                if _words(item) > cap:
                    msg = f"over limit: {head}[{i}] is {_words(item)} words (max {cap})"
                    violations.append(msg)
                    retryable.append(msg)
            continue
        value = data.get(path)
        if kind == "items" and isinstance(value, list) and len(value) > cap:
            violations.append(f"over limit: {path} had {len(value)} items (max {cap}), "
                              f"trimmed to {cap}")
            if trim:
                del value[cap:]
        elif kind == "words" and isinstance(value, str) and _words(value) > cap:
            msg = f"over limit: {path} is {_words(value)} words (max {cap})"
            violations.append(msg)
            retryable.append(msg)
    return violations, retryable


def specialist_payload(bundle: dict, name: str) -> dict:
    """The payload one specialist agent receives. Shared with the resume path."""
    return {
        "symbol": bundle["symbol"],
        "market": bundle["market"],
        "pillars": bundle["pillars"],
        "context": bundle.get("context", {}),
        "extras": bundle.get("extras", {}),
        "pillar_coverage": bundle.get("pillar_coverage", {}),
        "data_quality": bundle.get("data_quality", {}),
        "focus": {
            "fundamentals": ["fundamentals"],
            "valuation": ["valuation"],
            "technicals": ["technicals"],
            "risk": ["risk"],
            "governance": ["governance"],
        }.get(name, []),
    }


def audit_result(name: str, data: dict, payload: dict | None = None,
                 trim: bool = True) -> tuple[list[str], list[str]]:
    """The protocol checks for one agent output: key paths and output limits.

    Run on every fresh response, and again on anything restored by `--resume`. A
    resumed result is re-checked rather than trusted: stored before these checks
    existed, a 12 argument bull case would ship as compliant with an empty violation
    list, and it would enlarge every payload built from it.
    """
    limits = LIMITS_BY_AGENT.get((name or "").lower(), SPECIALIST_LIMITS)
    violations, retryable = check_limits(data, limits, trim=trim)
    return _check_citations(data, payload or {}) + violations, retryable


def bull_view(bull) -> dict:
    """The bull case as the bear sees it: the protocol caps, nothing else."""
    data = bull.data if hasattr(bull, "data") else (bull or {})
    args = (data.get("arguments") or [])[:BULL_ARGUMENT_CAP]
    return {
        "role": "bull",
        "thesis": data.get("thesis"),
        "arguments": [{"claim": a.get("claim"), "evidence": a.get("evidence")}
                      for a in args if isinstance(a, dict)],
        "what_would_break_this": (data.get("what_would_break_this")
                                  or [])[:BREAK_CONDITION_CAP],
    }


# The five pillars the bundle always reports coverage for. Citing one of these is
# legitimate provenance, so "pillar_coverage.governance" is accepted even when a
# test fixture omits the block. Any other sub-field is not a real key.
COVERAGE_FIELDS = ("fundamentals", "valuation", "technicals", "risk", "governance")


INDEX_RE = re.compile(r"\[(\d+)\]")


def _bundle_paths(bundle: dict) -> tuple[set[str], dict[str, int]]:
    """Every key path that actually exists in the bundle, plus the length of every list.

    Returns (paths, lists). `paths` are canonical and index-free, so `extras.
    institutional_holders.value` covers all eight holder rows. `lists` maps a list path
    to its length, which is what lets a *cited* index be checked against reality.

    This replaced a flat allowlist plus prefix matching. The prefix check
    (`k.startswith(("extras.", "context.", ...))`) accepted any invented key under
    those roots, so `context.peer_median_pe` or `extras.analyst_consensus_eps` - the
    exact peer/consensus data the bundle documents as absent - passed citation
    checking while being fabricated. A key is now valid only if the path resolves
    to something the bundle really contains.
    """
    paths: set[str] = set()
    lists: dict[str, int] = {}

    def walk(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else str(k)
                paths.add(path)
                walk(path, v)
        elif isinstance(obj, (list, tuple)):
            lists[prefix] = len(obj)
            for item in obj:
                walk(prefix, item)

    # Pillar values are cited as "group.key" (governance.beta), not "pillars.group.key",
    # which is what _normalise_key strips toward, so the prefix is dropped here.
    for group, vals in (bundle.get("pillars") or {}).items():
        if not isinstance(vals, dict):
            continue
        for k, v in vals.items():
            path = f"{group}.{k}"
            paths.add(path)
            walk(path, v)
    for top in ("context", "extras", "pillar_coverage", "data_quality"):
        block = bundle.get(top)
        if isinstance(block, dict):
            for k, v in block.items():
                path = f"{top}.{k}"
                paths.add(path)
                walk(path, v)
        elif block is not None:
            paths.add(top)
    return paths, lists


def _normalise_key(key: str) -> str:
    """Agents cite keys in several honest ways; accept the variants that resolve."""
    k = str(key).strip().strip('"')
    for prefix in ("pillars.", "$.pillars.", "bundle.pillars.", "evidence."):
        if k.startswith(prefix):
            return k[len(prefix):]
    # NOTE: "pillar_coverage.*" is deliberately NOT stripped. Stripping it would
    # turn "pillar_coverage.governance" into "governance", which then matches a
    # bare-signal-name check and silently validates a nonexistent key.
    return k


def _indices_resolve(key: str, lists: dict[str, int]) -> bool:
    """Every [i] in a cited key must address a row that actually exists.

    Paths are generated index-free, so without this check
    `extras.institutional_holders[99].value` resolves to the generated
    `extras.institutional_holders.value` and is accepted even though the bundle holds
    eight holders. Real agents do cite the indexed form (eight times in
    runs/AAPL_final.json), so the index has to be verified rather than stripped.
    """
    for m in INDEX_RE.finditer(key):
        prefix = re.sub(r"\[\d+\]", "", key[: m.start()]).strip(".")
        size = lists.get(prefix)
        if size is None or int(m.group(1)) >= size:
            return False
    return True


def _check_citations(data: dict, bundle: dict) -> list[str]:
    allowed, lists = _bundle_paths(bundle)
    # Bare names are accepted only one level below their root: "beta", "market_cap" or
    # "news_count" name a metric or field. Leaves nested inside a list item ("value",
    # "holder") do not, so accepting them would let "value" stand in for a citation.
    bare = {a.split(".")[-1] for a in allowed if a.count(".") == 1}
    cited = data.get("cited_metrics") or data.get("key_metrics") or []
    if isinstance(cited, str):
        # A single key, not a list of characters.
        cited = [cited]
    elif isinstance(cited, dict):
        cited = list(cited.keys()) + [v for v in cited.values() if isinstance(v, str)]
    # The debate agents cite a path per item, in a field the citation list does not
    # cover: "evidence" beside each bull argument and each bear attack, plus the bear's
    # strongest metric. Checked against the same allowlist, because an unchecked key
    # path is how a fabricated provenance reads as rigorous. A missing or malformed
    # path is a violation in its own right: skipping it would leave an unsupported
    # claim uninspected, which is the failure the check exists to prevent.
    cited = list(cited)
    bad: list[str] = []
    for field in ("arguments", "attacks"):
        for i, item in enumerate(data.get(field) or []):
            path = item.get("evidence") if isinstance(item, dict) else None
            if isinstance(path, str) and path.strip():
                cited.append(path)
            else:
                bad.append(f"missing evidence key: {field}[{i}]")
    strongest = data.get("strongest_bear_metric")
    if isinstance(strongest, dict):
        key = strongest.get("key")
        if isinstance(key, str) and key.strip():
            cited.append(key)
        else:
            bad.append("missing evidence key: strongest_bear_metric.key")
    for c in cited:
        if not isinstance(c, str):
            continue
        k = _normalise_key(c)
        if INDEX_RE.search(k) and not _indices_resolve(k, lists):
            bad.append(f"cited list index does not resolve: {c}")
            continue
        stripped = re.sub(r"\[\d+\]", "", k)          # extras.holders[0].value
        if stripped in allowed or ("." not in k and k in bare):
            continue
        if k.startswith("pillar_coverage.") and k.split(".", 1)[1] in COVERAGE_FIELDS:
            continue
        bad.append(f"uncited/unknown metric key: {c}")
    return bad


def call_agent(name: str, system: str, payload: dict, model: str = DEFAULT_MODEL,
               max_tokens: int = 1200, retries: int = 2, strict_json: bool = True) -> AgentResult:
    """Call one agent. strict_json forces the provider's JSON mode.

    Truncation is treated as a hard failure, not a partial success: a half-written
    JSON body is exactly the kind of plausible-looking output that must never be
    promoted to an analysis result.
    """
    user = json.dumps(payload, indent=1, default=str)[:14000]
    last_err = ""
    best: AgentResult | None = None
    correction = ""
    corrected = False
    for attempt in range(retries + 1):
        try:
            kwargs = {"response_format": {"type": "json_object"}} if strict_json else {}
            r = client().chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user + correction}],
                max_tokens=max_tokens,
                temperature=0.2,
                **kwargs,
            )
            choice = r.choices[0]
            raw = choice.message.content or ""
            tokens = r.usage.total_tokens if r.usage else 0
            if choice.finish_reason == "length":
                last_err = f"output truncated at max_tokens={max_tokens} (~{tokens} tokens used)"
                max_tokens = int(max_tokens * 1.8)  # widen once, then retry
                continue
            data = _extract_json(raw)
            if data is None:
                last_err = f"unparseable JSON on attempt {attempt + 1} (finish={choice.finish_reason})"
                continue
            data = strip_em_dashes(data)
            violations, retryable = audit_result(name, data, payload)
            result = AgentResult(
                name=name, status="ok", data=data, raw=raw, model=model, tokens=tokens,
                violations=violations,
            )
            # Keep the cleanest attempt. A result with no limit violation is returned at
            # once. A word overrun earns exactly one corrective re-call with the broken
            # limits restated, and after that the attempt with the fewest violations
            # wins, which is not always the last one.
            if best is None or len(result.violations) < len(best.violations):
                best = result
            if not violations:
                return result
            if retryable and not corrected and attempt < retries:
                corrected = True
                correction = ("\n\nCORRECTION from the pipeline: the previous answer broke "
                              "the stated output limits: " + "; ".join(retryable) +
                              ". Return the same analysis again, inside every limit. Cut "
                              "filler and repetition, not substance.")
                continue
            return best
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
    if best is not None:
        return best
    return AgentResult(name=name, status="unavailable", data={}, error=last_err, model=model)


def call_with_fallback(name: str, system: str, payload: dict, model: str = DEFAULT_MODEL,
                       max_tokens: int = 1600, **kw) -> AgentResult:
    """Primary model, then fallback model. Never silently returns a blank result."""
    res = call_agent(name, system, payload, model=model, max_tokens=max_tokens, **kw)
    if res.status == "ok" or not FALLBACK_MODEL or FALLBACK_MODEL == model:
        return res
    alt = call_agent(name, system, payload, model=FALLBACK_MODEL, max_tokens=max_tokens, **kw)
    if alt.status == "ok":
        alt.error = f"primary {model} failed: {res.error}"
    else:
        alt.error = f"primary({model}): {res.error} | fallback({FALLBACK_MODEL}): {alt.error}"
    return alt


def run_specialists(bundle: dict, model: str = DEFAULT_MODEL, workers: int = 9) -> dict:
    def one(name: str) -> AgentResult:
        payload = specialist_payload(bundle, name)
        return call_with_fallback(
            name,
            AGENT_SYSTEM.format(name=name, remit=SPECIALISTS[name]),
            payload, model=model, max_tokens=1300,
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, SPECIALISTS.keys()))
    return {r.name: r for r in results}


def _compact_bundle(bundle: dict, keep_extras: bool = False) -> dict:
    """Slim evidence package for the debate stages.

    Sending the full bundle plus the whole specialist digest pushed the bear over
    its token ceiling on both models. Keep only non-null pillar values.
    """
    pillars = {
        g: {k: v for k, v in vals.items() if v is not None}
        for g, vals in bundle["pillars"].items()
    }
    out = {
        "symbol": bundle["symbol"],
        "market": bundle["market"],
        "pillars": pillars,
        "pillar_coverage": bundle.get("pillar_coverage", {}),
        "context": bundle.get("context", {}),
        "data_quality": bundle.get("data_quality", {}),
    }
    if keep_extras:
        ex = dict(bundle.get("extras", {}))
        ex.pop("news_headlines", None)
        ex.pop("institutional_holders", None)
        out["extras"] = ex
    return out


def run_debate(bundle: dict, specialists: dict, model: str = DEFAULT_MODEL) -> dict:
    digest = {
        n: {k: v for k, v in r.data.items() if k in ("score", "confidence", "summary", "key_points")}
        for n, r in specialists.items() if r.status == "ok"
    }
    base = _compact_bundle(bundle)

    bull = call_with_fallback(
        "bull", BULL_SYSTEM,
        {**base, "specialist_views": digest},
        model=model, max_tokens=2600,
    )

    # The bear gets the bull case but not the specialist digest: it attacks the bull,
    # and digest noise was what pushed it past the token ceiling. The case is trimmed to
    # the protocol caps and to claim plus key path, because the bear's output is
    # proportional to its input, which is what makes the adversarial agent fail first.
    bear_case = bull_view(bull) if bull.status == "ok" else {}
    bear_payload = {**base, "bull_case": bear_case}
    bear = call_with_fallback("bear", BEAR_SYSTEM, bear_payload, model=model, max_tokens=3200)

    return {"bull": bull, "bear": bear}


def run_judge(bundle: dict, specialists: dict, debate: dict, verification: dict,
              horizon_scores: dict, model: str = DEFAULT_MODEL) -> AgentResult:
    payload = {
        "symbol": bundle["symbol"],
        "specialist_scores": {
            n: {"score": r.data.get("score"), "confidence": r.data.get("confidence"),
                "summary": r.data.get("summary")}
            for n, r in specialists.items() if r.status == "ok"
        },
        "failed_agents": [n for n, r in specialists.items() if r.status != "ok"],
        "bull": debate["bull"].data,
        "bear": debate["bear"].data,
        "verification": {k: v for k, v in verification.items() if k != "unverified_details"},
        "data_quality": bundle.get("data_quality", {}),
        "deterministic_scores": horizon_scores,
        "instruction": "The deterministic rubric verdict is final. Explain it. Do not override it.",
    }
    return call_with_fallback("judge", JUDGE_SYSTEM, payload, model=model, max_tokens=2600, retries=1)
