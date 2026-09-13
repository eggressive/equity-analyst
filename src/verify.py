"""Verification layer. Independent of the agents that produced the text.

The article's core claim is that every claim must be traceable to a source, and
that the Bear gets to attack the Bull before a score is issued. This module does
the mechanical part of that: it checks whether the numbers an agent quoted
actually exist in the evidence bundle, and flags anything it cannot match.

Scope, stated plainly, because this is weaker than the name suggests. Matching is
value-level, not claim-level: a number is accepted if it equals an evidence value
within `rel_tol`, or a ratio, difference or percentage change between two related
evidence values whose metrics are named near the number. The check does not verify
that a number was attached to the metric the agent claimed it for, so "net margin is
18.0%" still passes when 18.0 is a P/E elsewhere in the bundle.

Measured on an AAPL bundle of 66 metric values plus coverage, context and news count,
2026-09-13: the derived index holds 3,648 values, and a random number in 0.05..100 is
accepted 27% of the time (36% when it looks like a percentage, 2.7% when it looks like
money), because percentages are dense. Before the attribution rule the same harness
accepted 100%. What attribution buys is the paragraph: a fully invented 12-number
paragraph returned PASS on 100% of draws, and on none of 2,000 draws now, with 8.6
ungrounded numbers left per paragraph. Read `unverified == 0` as "nothing obviously
invented", not as proof of grounding. Judge output is never passed through this module.
"""

from __future__ import annotations

import re

NUMBER_RE = re.compile(
    r"(?<![\w.])(-?\d[\d,]*\.?\d*)\s*(%|x|bn|billion|b|trn|tn|trillion|t|m|million|crore|cr|k|thousand)?(?![a-zA-Z0-9])",
    re.I,
)
# The trailing negative lookahead is load-bearing: without it "Beta of 1.08 means"
# parses as "1.08 million", because the optional unit group happily eats the "m"
# of "means". That produced a phantom unverified claim in every AAPL run.

# Phrases where the digits are a label, not a claim: "200-day average",
# "52-week high", "14-day RSI", "50/200-day averages", and ISO dates
# ("2026-10-29"). Counting their digits as claims both inflates the count and
# lets a fragment like 10 collide with an unrelated evidence value.
# `_mask_labels` replaces them with same-length blanks, so the offsets still
# address the original text for attribution.
LABEL_PHRASES = re.compile(
    r"\b\d{1,3}(?:\s*[-/,]\s*(?:and\s+)?\d{1,3})*\s*[- ]\s*(?:day|week|month|year|quarter|yr)s?\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    # A written date is a date too: "October 16" is a day of the month, not a claim.
    # The lookahead keeps "margin 25.58%" and "market 3" out of the mask.
    r"|\b(?:january|february|march|april|may|june|july|august|september|october|november"
    r"|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?(?![a-z])\s+"
    r"\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august"
    r"|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov"
    r"|dec)\.?(?![a-z])(?:\s+\d{4})?\b",
    re.I,
)

# A run passes when at most this share of its numbers is ungrounded, and a short
# text also fails on an absolute count so a few invented numbers cannot hide in a
# small sample.
PASS_UNVERIFIED_RATIO = 0.20
SHORT_TEXT_CLAIMS = 20
SHORT_TEXT_FLOOR = 2

# Percentage-stored metrics: `_pct` suffixed keys plus the three that carry a prefix
# instead. A fraction written as a percentage (0.375 and "37.5%") is one quantity;
# reading a percentage as its fraction is only allowed for these, because revenue
# 416.2B is not 4.162B and a PE of 40 is not 0.4x.
_PCT_KEY_RE = re.compile(r"(_pct$|^pct_|trend_vs_|percentage)", re.I)

UNITS = {
    "%": 1.0, "x": 1.0, "": 1.0,
    "bn": 1e9, "billion": 1e9, "b": 1e9,
    "trn": 1e12, "tn": 1e12, "trillion": 1e12, "t": 1e12,
    "m": 1e6, "million": 1e6, "crore": 1e7, "cr": 1e7,
    "k": 1e3, "thousand": 1e3,
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


def extract_numbers_at(text: str) -> list[tuple[float, str, int]]:
    """Numbers with their unit applied and their offset in `text`."""
    found = []
    for m in NUMBER_RE.finditer(text or ""):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        found.append((val * UNITS.get(unit, 1.0), unit, m.start(1)))
    return found


def extract_numbers(text: str) -> list[tuple[float, str]]:
    """Numbers only, for callers that do not need attribution."""
    return [(value, unit) for value, unit, _ in extract_numbers_at(text)]


def _mask_labels(text: str) -> str:
    """Blank out lookback-window phrases and dates, keeping the offsets."""
    return LABEL_PHRASES.sub(lambda m: " " * len(m.group(0)), text or "")


def _percent_named(key: str) -> bool:
    """Is this metric's value stored as a percentage rather than a fraction?"""
    return bool(_PCT_KEY_RE.search(key.rsplit(".", 1)[-1] or ""))


def _basis_named(named: set[str], ka: str, kb: str) -> bool:
    """Both metrics behind a derived value must be named beside the number.

    Naming one operand is not enough. With revenue and free cash flow in the bundle,
    "the figure is 2x" named nothing and "revenue doubled by 2x" named one operand,
    and both passed as an FCF/revenue derivation. A number with no metric named near
    it is still accepted when it equals an evidence value: only the derived index
    needs its basis spelled out.
    """
    return ka in named and kb in named


# A derived value is credible only when the metrics behind it are related: same
# pillar, or one of them a price or level. Market cap minus revenue growth is
# arithmetic nobody does; free cash flow over market cap is a yield someone quotes.
LEVEL_KEYS = frozenset({
    "technicals.last_close", "technicals.sma50", "technicals.sma200",
    "technicals.high_52w", "technicals.low_52w",
    "valuation.market_cap", "valuation.fifty_day_avg", "valuation.two_hundred_day_avg",
})


def _related(ka: str, kb: str) -> bool:
    return (ka.split(".")[0] == kb.split(".")[0]) or ka in LEVEL_KEYS or kb in LEVEL_KEYS


def _derived_index(known: list[tuple[str, float]]) -> list[tuple[float, str, str, str, str]]:
    """Index of legitimate derived values: ratios, differences and percentage
    changes between two related evidence values.

    An agent that reports "equity is 1/45.15 of the price" or "the gap to the
    52-week high is 2.21%" is not fabricating: it is doing arithmetic on sourced
    numbers. Treating those as unverified would flag honest analysis, so they are
    matched separately and reported as derived rather than silently passed.

    The cap matters. Every unordered pair of unrelated metrics adds three more
    numbers to the acceptable set, which is how a 66-value bundle came to accept
    99.9% of random numbers between 0.05 and 100. Unrelated pairs are dropped.
    """
    derived = []
    vals = [(k, v) for k, v in known if v not in (None, 0)]
    for i, (ka, va) in enumerate(vals):
        for kb, vb in vals[i + 1:]:
            if not _related(ka, kb):
                continue
            derived.append((va / vb, f"{ka} / {kb}", ka, kb, "ratio"))
            # Both directions: "FCF is 85.7% of net income" needs fcf / net_income
            # even when net_income comes first in the bundle.
            derived.append((vb / va, f"{kb} / {ka}", ka, kb, "ratio"))
            derived.append((va - vb, f"{ka} - {kb}", ka, kb, "difference"))
            derived.append(((va - vb) / vb * 100, f"({ka} - {kb}) / {kb} pct", ka, kb, "pct"))
    return derived


# Metric names as prose: attribution needs to know which metric a number belongs
# to, and agents write "net margin", not "net_margin_pct". Extend this table when
# a metric starts being cited by a name it does not cover.
METRIC_ALIASES = {
    "fundamentals.revenue": ["revenue", "sales", "top line"],
    "fundamentals.revenue_growth_yoy_pct": ["revenue growth", "sales growth",
                                            "top-line growth"],
    "fundamentals.net_income": ["net income", "net earnings", "ni"],
    "fundamentals.gross_margin_pct": ["gross margin"],
    "fundamentals.operating_margin_pct": ["operating margin"],
    "fundamentals.net_margin_pct": ["net margin", "profit margin"],
    "fundamentals.fcf": ["free cash flow", "fcf"],
    "fundamentals.fcf_margin_pct": ["fcf margin", "free cash flow margin"],
    "fundamentals.cash_conversion_ocf_over_ni": ["cash conversion", "ocf/ni", "ocf / ni"],
    "fundamentals.debt_to_equity": ["debt to equity", "debt-to-equity", "debt/equity", "d/e"],
    "fundamentals.roa_pct": ["roa", "return on assets"],
    "fundamentals.roe_pct": ["roe", "return on equity"],
    "fundamentals.cash_to_assets_pct": ["cash to assets", "cash/assets"],
    "valuation.pe_ttm": ["p/e", "pe ratio", "price to earnings", "price-to-earnings",
                         "ttm earnings", "trailing earnings"],
    "valuation.pe_forward": ["forward p/e", "forward pe", "forward earnings"],
    "valuation.pb": ["p/b", "price to book", "price-to-book", "book value", "book"],
    "valuation.ps_ttm": ["p/s", "price to sales", "price-to-sales"],
    "valuation.ev_to_ebitda": ["ev/ebitda", "ev to ebitda"],
    "valuation.ev_to_revenue": ["ev/revenue", "ev to revenue"],
    "valuation.peg": ["peg"],
    "valuation.dividend_yield_pct": ["dividend yield", "dividend"],
    "valuation.market_cap": ["market cap", "market capitalisation", "market capitalization"],
    "valuation.fifty_day_avg": ["50-day average", "50 day average"],
    "valuation.two_hundred_day_avg": ["200-day average", "200 day average"],
    "technicals.last_close": ["share price", "last close", "trading at", "price"],
    "technicals.sma50": ["sma50", "50-day moving average"],
    "technicals.sma200": ["sma200", "200-day moving average"],
    "technicals.rsi14": ["rsi"],
    "technicals.atr14_pct": ["atr"],
    "technicals.vol_ann_pct": ["annualised volatility", "annualized volatility", "volatility"],
    "technicals.max_drawdown_pct": ["max drawdown", "drawdown", "peak to trough",
                                    "peak-to-trough"],
    "technicals.high_52w": ["52-week high", "52 week high", "52w high"],
    "technicals.low_52w": ["52-week low", "52 week low", "52w low"],
    "technicals.pct_from_52w_high": ["52-week high", "52 week high", "52w high"],
    "technicals.trend_vs_sma50": ["50-day average", "50-day", "above the 50-day"],
    "technicals.trend_vs_sma200": ["200-day average", "200-day", "above the 200-day"],
    "technicals.volume_vs_20d_avg": ["20-day average volume", "volume versus", "volume vs"],
    "risk.beta": ["beta"],
    "risk.annualised_vol_pct": ["volatility"],
    "risk.max_drawdown_pct": ["max drawdown", "drawdown"],
    "risk.downside_deviation_pct": ["downside deviation"],
    "risk.held_by_institutions_pct": ["institutions", "institutional"],
    "risk.short_ratio": ["short ratio", "days to cover"],
    "governance.held_by_institutions_pct": ["institutions", "institutional"],
    "governance.held_by_insiders_pct": ["insiders", "insider"],
    "governance.share_dilution_pct": ["share dilution", "dilution", "share count", "buyback"],
    "governance.audit_risk": ["audit risk"],
    "governance.overall_risk": ["overall risk"],
    "governance.compensation_risk": ["compensation risk"],
    "governance.shares_outstanding": ["shares outstanding"],
}

_ALIAS_RES = {
    key: re.compile("|".join(r"(?<![\w])" + re.escape(a) + r"(?![\w])"
                             for a in sorted(aliases, key=len, reverse=True)), re.I)
    for key, aliases in METRIC_ALIASES.items()
}


def named_metrics(text: str, pos: int, span: int = 80) -> set[str]:
    """Metric keys named in the sentence or neighbourhood holding offset `pos`.

    Longest alias wins over the aliases it contains, so "revenue growth" names
    revenue_growth_yoy_pct without also naming revenue.
    """
    if not text:
        return set()
    left = max(text.rfind(ch, 0, pos) for ch in ".;\n")
    rights = [h for h in (text.find(ch, pos) for ch in ".;\n") if h != -1]
    right = min(rights) if rights else len(text)
    if right - left < 40:
        left, right = max(0, pos - span), min(len(text), pos + span)
    window = text[left:right]
    hits = []
    for key, pat in _ALIAS_RES.items():
        for m in pat.finditer(window):
            hits.append((m.start(), m.end(), m.end() - m.start(), key))
    hits.sort(key=lambda hit: -hit[2])
    taken, out = [], set()
    for start, end, _, key in hits:
        if any(not (end <= ts or start >= te) for ts, te in taken):
            continue
        taken.append((start, end))
        out.add(key)
    return out


def verify_claims(text: str, pillars: dict, rel_tol: float = 0.02,
                  declared_scores: list[float] | None = None,
                  extras: dict | None = None) -> dict:
    """Check every number in an agent's output against the evidence bundle.

    A number is accepted when it matches an evidence value, or a derived value
    whose two metrics are named near the number. Sign is not checked: "shares fell
    8.03%" states the magnitude of a metric recorded as -8.03.
    """
    known = _flatten(pillars, extras=extras)
    derived = _derived_index(known)
    # Agents also report their own -2..+2 judgment scores. Those are opinions the
    # protocol asked for, not evidence claims, so accept them explicitly instead
    # of leaving them to collide with unrelated metric values.
    if declared_scores:
        known = known + [(f"agent_score[{s}]", float(s)) for s in declared_scores]
    body = text or ""
    numbers = extract_numbers_at(_mask_labels(body))

    def close(value: float, target: float) -> bool:
        if target == 0:
            return value == 0
        return abs(value - target) <= abs(target) * rel_tol or \
            abs(value + target) <= abs(target) * rel_tol

    def candidates(value: float, unit: str, target: float,
                   divide_ok: bool = False) -> list[float]:
        """The target, plus its percentage or fractional twin.

        "Governance coverage is 37.5%" and the bundle's 0.375 are the same quantity,
        and so are "85.7% of net income" and the ratio 0.857. Scale-bridging keeps
        the index small; generating both forms would double it. The reverse bridge,
        reading a percentage as its fraction, needs the metric to be a percentage.
        """
        out = [target]
        if unit == "%" and 0 < abs(target) < 1:
            out.append(target * 100)
        elif unit in ("", "x") and abs(target) > 1 and divide_ok:
            out.append(target / 100)
        return out

    def match(value: float, unit: str, start: int):
        named = named_metrics(body, start)
        for key, kv in known:
            if any(close(value, c) for c in candidates(value, unit, kv,
                                                       divide_ok=_percent_named(key))):
                return ("evidence", key)
        if value == 0:
            return (None, None)
        for dv, expr, ka, kb, kind in derived:
            if not _basis_named(named, ka, kb):
                continue
            if any(close(value, c) for c in candidates(value, unit, dv,
                                                       divide_ok=kind == "pct")):
                return ("derived", expr)
        return (None, None)

    verified, derived_hits, unverified = [], [], []
    for value, unit, start in numbers:
        # A bare 4-digit year is a date, not a financial claim. Counting it as an
        # ungrounded number would make every honest analysis look unreliable.
        if not unit and float(value).is_integer() and 1900 <= value <= 2100:
            continue
        kind, ref = match(value, unit, start)
        row = {"value": value, "unit": unit, "matched_metric": ref, "match_type": kind,
               "at": start}
        if kind == "evidence":
            verified.append(row)
        elif kind == "derived":
            derived_hits.append(row)
        else:
            unverified.append(row)

    total = len(verified) + len(derived_hits) + len(unverified)
    grounded = len(verified) + len(derived_hits)
    # The ratio alone is too generous for a short text: 12 claims pass with two
    # ungrounded numbers, which is 16% of a paragraph and none of its substance.
    # Short texts need the absolute floor as well.
    failed = total > 0 and len(unverified) / total > PASS_UNVERIFIED_RATIO
    if total and total <= SHORT_TEXT_CLAIMS and len(unverified) >= SHORT_TEXT_FLOOR:
        failed = True
    return {
        "claims_found": total,
        "verified": len(verified),
        "derived_from_evidence": len(derived_hits),
        "unverified": len(unverified),
        "grounding_rate": round(grounded / total, 3) if total else None,
        "unverified_details": unverified[:20],
        "unverified_at": [row["at"] for row in unverified],
        "verdict": "PASS" if total and not failed else "REVIEW" if total else "NO_CLAIMS",
    }
