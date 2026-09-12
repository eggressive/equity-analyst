"""Verification layer. Independent of the agents that produced the text.

The article's core claim is that every claim must be traceable to a source, and
that the Bear gets to attack the Bull before a score is issued. This module does
the mechanical part of that: it checks whether the numbers an agent quoted
actually exist in the evidence bundle, and flags anything it cannot match.

A claim is only accepted if the number appears in the allowed evidence values
(relative tolerance) AND the pillar it was attributed to is present.
"""

from __future__ import annotations

import re

NUMBER_RE = re.compile(
    r"(?<![\w.])(-?\d[\d,]*\.?\d*)\s*(%|x|bn|billion|m|million|crore|cr)?(?![a-zA-Z0-9])",
    re.I,
)
# The trailing negative lookahead is load-bearing: without it "Beta of 1.08 means"
# parses as "1.08 million", because the optional unit group happily eats the "m"
# of "means". That produced a phantom unverified claim in every AAPL run.

# Phrases where the digits are a label, not a claim: "200-day average",
# "52-week high", "14-day RSI". Counting them as ungrounded numbers penalises
# agents for naming the window they used.
LABEL_PHRASES = re.compile(
    r"\b\d{1,3}\s*[- ]\s*(?:day|week|month|year|quarter|yr)s?\b", re.I
)

UNITS = {
    "%": 1.0, "x": 1.0, "": 1.0, "bn": 1e9, "billion": 1e9,
    "m": 1e6, "million": 1e6, "crore": 1e7, "cr": 1e7,
}


def _flatten(pillars: dict, extras: dict | None = None) -> list[tuple[str, float]]:
    """Collect every numeric value an agent is allowed to cite.

    Includes nested extras (e.g. institutional holder amounts) so that sourced
    figures outside the five pillars are not reported as ungrounded.
    """
    out: list[tuple[str, float]] = []

    def walk(prefix: str, obj):
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            out.append((prefix, float(obj)))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}", v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(f"{prefix}[{i}]", v)

    for pillar, vals in (pillars or {}).items():
        for k, v in vals.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((f"{pillar}.{k}", float(v)))
    if extras:
        walk("extras", extras)
    return out


def extract_numbers(text: str) -> list[tuple[float, str]]:
    found = []
    for m in NUMBER_RE.finditer(text or ""):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        found.append((val * UNITS.get(unit, 1.0), unit))
    return found


def _strip_labels(text: str) -> str:
    """Remove lookback-window phrases before extracting numbers."""
    return LABEL_PHRASES.sub(" <window> ", text or "")


def _derived_index(known: list[tuple[str, float]]) -> list[tuple[float, str]]:
    """Index of legitimate derived values: ratios, differences and percentage
    changes between two evidence values.

    An agent that reports "equity is 1/45.15 of the price" or "the gap to the
    52-week high is 2.21%" is not fabricating: it is doing arithmetic on sourced
    numbers. Treating those as unverified would flag honest analysis, so they are
    matched separately and reported as derived rather than silently passed.
    """
    derived = []
    vals = [(k, v) for k, v in known if v not in (None, 0)]
    for ka, va in vals:
        for kb, vb in vals:
            if ka >= kb:
                continue
            derived.append((va / vb, f"{ka} / {kb}"))
            derived.append(((va - vb) / vb * 100, f"({ka} - {kb}) / {kb} pct"))
            derived.append((va - vb, f"{ka} - {kb}"))
    return derived


def verify_claims(text: str, pillars: dict, rel_tol: float = 0.02,
                  declared_scores: list[float] | None = None,
                  extras: dict | None = None) -> dict:
    """Check every number in an agent's output against the evidence bundle."""
    known = _flatten(pillars, extras=extras)
    derived = _derived_index(known)
    # Agents also report their own -2..+2 judgment scores. Those are opinions the
    # protocol asked for, not evidence claims, so accept them explicitly instead
    # of leaving them to collide with unrelated metric values.
    if declared_scores:
        known = known + [(f"agent_score[{s}]", float(s)) for s in declared_scores]
    numbers = extract_numbers(_strip_labels(text))

    def match(value: float):
        for key, kv in known:
            if kv == 0:
                if value == 0:
                    return ("evidence", key)
                continue
            if abs(value - kv) <= abs(kv) * rel_tol:
                return ("evidence", key)
        if value == 0:
            return (None, None)
        for dv, expr in derived:
            if abs(value - dv) <= abs(dv) * rel_tol:
                return ("derived", expr)
        return (None, None)

    verified, derived_hits, unverified = [], [], []
    for value, unit in numbers:
        # A bare 4-digit year is a date, not a financial claim. Counting it as an
        # ungrounded number would make every honest analysis look unreliable.
        if not unit and float(value).is_integer() and 1900 <= value <= 2100:
            continue
        kind, ref = match(value)
        row = {"value": value, "unit": unit, "matched_metric": ref, "match_type": kind}
        if kind == "evidence":
            verified.append(row)
        elif kind == "derived":
            derived_hits.append(row)
        else:
            unverified.append(row)

    total = len(numbers)
    grounded = len(verified) + len(derived_hits)
    return {
        "claims_found": total,
        "verified": len(verified),
        "derived_from_evidence": len(derived_hits),
        "unverified": len(unverified),
        "grounding_rate": round(grounded / total, 3) if total else None,
        "unverified_details": unverified[:20],
        "verdict": (
            "PASS" if total and len(unverified) / total <= 0.20
            else "REVIEW" if total else "NO_CLAIMS"
        ),
    }
