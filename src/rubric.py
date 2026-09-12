"""Deterministic scoring rubric. The model never decides the verdict; this does.

Design rules, all deliberate:
  1. Every signal is a pure function of an auditable number. No agent input.
  2. Signals score on a fixed -2..+2 scale with documented thresholds.
  3. Pillar score = mean of its *available* signals (missing signals do not score 0).
  4. Coverage = share of signals available. Low coverage dilutes weight instead of
     silently counting as neutral.
  5. Two horizons use different weights. SHORT_TERM leans on price/risk; LONG_TERM
     leans on fundamentals/valuation/governance.
  6. Overall confidence = weighted coverage. Below the floor the verdict is
     INSUFFICIENT_DATA rather than a guess.
"""

from __future__ import annotations


def _band(x: float) -> str:
    if x >= 1.0:
        return "STRONG_BULLISH"
    if x >= 0.35:
        return "BULLISH"
    if x > -0.35:
        return "NEUTRAL"
    if x > -1.0:
        return "BEARISH"
    return "STRONG_BEARISH"


def _s(x, thresh: list[tuple[float, int]]) -> int | None:
    """Map a number to -2..+2.

    Threshold rows MUST be written in ascending bound order with the semantics
    "if x < bound then this score". Direction is encoded in the table itself, never
    by a separate flag: an inversion flag plus an already-ordered table silently
    flips the sign of every signal (this shipped once and made low-debt companies
    score as highly leveraged).
    """
    if x is None:
        return None
    for limit, score in thresh:
        if x < limit:
            return score
    return thresh[-1][1] if thresh else 0


def _sig(pillars: dict, group: str, key: str):
    return pillars.get(group, {}).get(key)


def build_signals(p: dict) -> dict[str, dict[str, int | None]]:
    """Turn raw metrics into discrete, explainable signals.

    Every table below is ascending and reads "if x < bound -> score". Tables whose
    metric is better when LOW (debt, multiples, volatility, beta) therefore invert
    their scores by hand, e.g. [(0.3, 2), (0.6, 1), (1.0, 0), (1.5, -1), (99, -2)].
    """
    return {
        "fundamentals": {
            "revenue_growth": _s(_sig(p, "fundamentals", "revenue_growth_yoy_pct"),
                                 [(-5, -2), (0, -1), (5, 0), (15, 1), (25, 1)]),
            "net_margin": _s(_sig(p, "fundamentals", "net_margin_pct"),
                             [(0, -2), (5, -1), (10, 0), (20, 1), (99, 2)]),
            # Operating margin must be its own signal: it is the cleanest read on
            # core-business profitability and is cited by agents, but if it is not
            # in this table a wrong value silently fails to move the verdict.
            "operating_margin": _s(_sig(p, "fundamentals", "operating_margin_pct"),
                                   [(0, -2), (5, -1), (12, 0), (20, 1), (999, 2)]),
            "fcf_margin": _s(_sig(p, "fundamentals", "fcf_margin_pct"),
                             [(0, -2), (5, -1), (10, 0), (20, 1), (99, 2)]),
            "cash_conversion": _s(_sig(p, "fundamentals", "cash_conversion_ocf_over_ni"),
                                  [(0.7, -2), (0.9, -1), (1.1, 0), (1.4, 1), (99, 1)]),
            # lower is better
            "leverage": _s(_sig(p, "fundamentals", "debt_to_equity"),
                           [(0.3, 2), (0.6, 1), (1.0, 0), (1.5, -1), (99, -2)]),
            "roe": _s(_sig(p, "fundamentals", "roe_pct"),
                      [(0, -2), (8, -1), (15, 0), (25, 1), (999, 2)]),
        },
        "valuation": {
            # lower is better for all multiples
            "pe_ttm": _s(_sig(p, "valuation", "pe_ttm"), [(0, -1), (15, 2), (25, 1), (40, 0), (60, -1), (9999, -2)]),
            "pe_forward": _s(_sig(p, "valuation", "pe_forward"), [(0, -1), (15, 2), (25, 1), (40, 0), (60, -1), (9999, -2)]),
            "pb": _s(_sig(p, "valuation", "pb"), [(0, -1), (2, 2), (4, 1), (8, 0), (15, -1), (9999, -2)]),
            "ev_to_ebitda": _s(_sig(p, "valuation", "ev_to_ebitda"), [(0, -1), (10, 2), (15, 1), (22, 0), (35, -1), (9999, -2)]),
            "ps_ttm": _s(_sig(p, "valuation", "ps_ttm"), [(0, -1), (2, 2), (4, 1), (8, 0), (15, -1), (9999, -2)]),
        },
        "technicals": {
            "trend_sma200": _s(_sig(p, "technicals", "trend_vs_sma200"), [(-10, -2), (-2, -1), (2, 0), (10, 1), (999, 1)]),
            "trend_sma50": _s(_sig(p, "technicals", "trend_vs_sma50"), [(-8, -2), (-2, -1), (2, 0), (8, 1), (999, 1)]),
            # RSI is a mean-reversion input: oversold favours entry, overbought is caution.
            "rsi14": _s(_sig(p, "technicals", "rsi14"), [(25, 1), (35, 1), (55, 0), (70, -1), (200, -1)]),
            "from_52w_high": _s(_sig(p, "technicals", "pct_from_52w_high"), [(-40, 1), (-20, 0), (-8, -1), (0, -1)]),
        },
        "risk": {
            # lower is better for all three
            "volatility": _s(_sig(p, "risk", "annualised_vol_pct"), [(20, 2), (30, 1), (45, 0), (60, -1), (999, -2)]),
            "max_drawdown": _s(_sig(p, "risk", "max_drawdown_pct"), [(-50, -1), (-30, 0), (-15, 1), (0, 2)]),
            "beta": _s(_sig(p, "risk", "beta"), [(0.8, 2), (1.1, 1), (1.4, 0), (2.0, -1), (99, -2)]),
        },
        "governance": {
            "institutional_holding": _s(_sig(p, "governance", "held_by_institutions_pct"),
                                        [(5, -1), (25, 0), (50, 1), (101, 1)]),
            "insider_holding": _s(_sig(p, "governance", "held_by_insiders_pct"),
                                  [(0, -2), (1, -1), (5, 0), (15, 1), (100, 1)]),
            "dilution": _s(_sig(p, "governance", "share_dilution_pct"),
                           [(-3, 2), (-1, 1), (1, 0), (3, -1), (9999, -2)]),
        },
    }


WEIGHTS = {
    "SHORT_TERM": {"fundamentals": 0.10, "valuation": 0.15, "technicals": 0.35, "risk": 0.30, "governance": 0.10},
    "LONG_TERM":  {"fundamentals": 0.32, "valuation": 0.26, "technicals": 0.07, "risk": 0.15, "governance": 0.20},
}

COVERAGE_FLOOR = 0.45


def score(signals: dict, horizon: str) -> dict:
    weights = WEIGHTS[horizon]
    pillar_out = {}
    weighted_sum = 0.0
    weight_used = 0.0

    for pillar, weight in weights.items():
        vals = [v for v in signals.get(pillar, {}).values() if v is not None]
        total = len(signals.get(pillar, {}))
        coverage = (len(vals) / total) if total else 0.0
        score_val = (sum(vals) / len(vals)) if vals else None
        pillar_out[pillar] = {
            "score": None if score_val is None else round(score_val, 3),
            "coverage": round(coverage, 3),
            "weight": weight,
            "signals": {k: v for k, v in signals.get(pillar, {}).items() if v is not None},
            "missing": [k for k, v in signals.get(pillar, {}).items() if v is None],
        }
        if score_val is not None:
            weighted_sum += weight * score_val * coverage  # coverage dilutes
            weight_used += weight * coverage

    overall = round(weighted_sum / weight_used, 3) if weight_used else None
    confidence = round(weight_used / sum(weights.values()), 3)

    if confidence < COVERAGE_FLOOR or overall is None:
        verdict = "INSUFFICIENT_DATA"
    else:
        verdict = _band(overall)

    return {
        "horizon": horizon,
        "verdict": verdict,
        "overall_score": overall,
        "confidence": confidence,
        "pillars": pillar_out,
        "rubric": WEIGHTS[horizon],
        "rule": "overall = sum(weight * pillar_score * coverage) / sum(weight * coverage)",
    }
