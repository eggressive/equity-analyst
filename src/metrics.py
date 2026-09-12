"""Deterministic metric engine. Python calculates; agents only interpret.

Every number an agent receives comes from here. No agent is allowed to invent one.
Each pillar returns raw values only. Coverage is derived elsewhere: `pillar_coverage`
counts how many declared fields are non-null, and `rubric.score` derives its own
per-signal coverage, which is the one that dilutes weight.
"""

from __future__ import annotations

import math

import re

import pandas as pd


def _num(x):
    try:
        if x is None:
            return None
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text) -> list[str]:
    return _WORD.findall(str(text).lower())


def _pick_label(df: pd.DataFrame, *needles: str) -> str | None:
    """Find the statement row that best matches a needle.

    Naive substring matching is unsafe here: the label
    "Other Non Operating Income Expenses" contains "operating income", and it
    sorts earlier in the index than the real "Operating Income" row. That single
    bug reported Reliance's operating margin as 0.34% instead of 11.48% and the
    Bear built an entire (wrong) thesis on it.

    Rules:
      1. The needle must match whole words, so "EBIT" never matches "EBITDA".
      2. All needle tokens must be present.
      3. Among candidates, the fewest extra tokens wins, so "Operating Income"
         beats "Other Non Operating Income Expenses".
      4. An exact label match short-circuits everything.
    """
    if df is None or df.empty:
        return None
    labels = [(str(lbl), _tokens(lbl)) for lbl in df.index]

    for needle in needles:
        nt = _tokens(needle)
        if not nt:
            continue
        exact = [lbl for lbl, tk in labels if tk == nt]
        if exact:
            return exact[0]

        scored = []
        for lbl, tk in labels:
            if all(t in tk for t in nt):
                scored.append((len(tk) - len(nt), len(tk), lbl))
        if scored:
            scored.sort()
            return scored[0][2]

        # fall back to subsequence-free containment on the raw string
        for lbl, _tk in labels:
            if needle.lower() in lbl.lower():
                return lbl
    return None


def _row(df: pd.DataFrame, *needles: str, col: int = 0):
    """Find the value of the best-matching row. Statements vary by market."""
    lbl = _pick_label(df, *needles)
    if lbl is None:
        return None
    try:
        return _num(df.loc[lbl].iloc[col])
    except Exception:
        return None


def _row_series(df: pd.DataFrame, *needles: str):
    lbl = _pick_label(df, *needles)
    if lbl is None:
        return None
    try:
        return pd.to_numeric(df.loc[lbl], errors="coerce")
    except Exception:
        return None


def _pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return round(a / b * 100, 2)


def _r(x, nd=2):
    return None if x is None else round(x, nd)


# ---------------------------------------------------------------- pillars


def fundamentals(b) -> dict:
    inc, bs, cf = b.income, b.balance, b.cashflow
    rev = _row(inc, "Total Revenue", "Operating Revenue", "Total Revenues")
    gp = _row(inc, "Gross Profit")
    op = _row(inc, "Operating Income", "EBIT")
    ni = _row(inc, "Net Income Common", "Net Income")
    rev_prev = _row(inc, "Total Revenue", "Operating Revenue", col=1) if inc.shape[1] > 1 else None

    debt = _row(bs, "Total Debt")
    if debt is None:
        # Fall back to components. Two traps live here, and both made a partially
        # reported balance sheet look less levered than it is:
        #   1. `or 0` turned "no long-term debt row" into a real zero, so a
        #      company whose debt rows are missing entirely scored as debt-free
        #      (leverage signal +2, the most bullish value on the table).
        #   2. The combined capital-lease row excludes current debt, and it used
        #      to overwrite the long+short sum computed above it.
        # Unknown stays None: absence must never score.
        # The "Long Term Debt" needle also matches the capital-lease variant, which
        # is the same long-term quantity for this purpose.
        long_term = _row(bs, "Long Term Debt")
        short_term = _row(bs, "Current Debt", "Short Long Term Debt",
                          "Current Debt And Capital Lease Obligation")
        if long_term is not None or short_term is not None:
            debt = (long_term or 0) + (short_term or 0)
    equity = _row(bs, "Stockholders Equity", "Total Equity Gross Minority", "Common Stock Equity")
    assets = _row(bs, "Total Assets")
    cash = _row(bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    ocf = _row(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex = _row(cf, "Capital Expenditure")
    fcf = _row(cf, "Free Cash Flow")
    if fcf is None and ocf is not None and capex is not None:
        fcf = ocf + capex  # capex is negative in these statements

    out = {
        "revenue": rev,
        "revenue_growth_yoy_pct": _pct(None if rev is None or rev_prev is None else rev - rev_prev, rev_prev),
        "net_income": ni,
        "gross_margin_pct": _pct(gp, rev),
        "operating_margin_pct": _pct(op, rev),
        "net_margin_pct": _pct(ni, rev),
        "fcf": fcf,
        "fcf_margin_pct": _pct(fcf, rev),
        "cash_conversion_ocf_over_ni": _r(None if ocf is None or ni in (None, 0) else ocf / ni),
        "debt_to_equity": _r(None if debt is None or equity in (None, 0) else debt / equity),
        "net_debt_to_ebitda": None,
        "roa_pct": _pct(ni, assets),
        "roe_pct": _pct(ni, equity),
        "cash_to_assets_pct": _pct(cash, assets),
    }
    return out


def _dividend_yield_pct(q: dict):
    """Dividend yield as a percent, without double-scaling.

    yfinance 1.7 returns `dividendYield` already scaled as a percent (AAPL reports
    0.33 for a ~0.33% yield), while older builds returned a fraction (0.0033). The
    code used to multiply by 100 unconditionally, so AAPL's yield was published as
    33.0% and the agents spent a paragraph each treating it as a data defect.

    When `dividendRate` and a price are both present, dividendRate/price decides
    which reading of `dividendYield` is the consistent one. With no cross-check,
    the pinned provider's semantics (already a percent) are used.
    """
    raw = _num(q.get("dividendYield"))
    if raw is None:
        return None
    rate = _num(q.get("dividendRate"))
    price = (_num(q.get("regularMarketPrice")) or _num(q.get("currentPrice"))
             or _num(q.get("previousClose")))
    if rate is not None and price:
        implied = rate / price * 100
        if implied > 0:
            as_percent = abs(raw - implied)
            as_fraction = abs(raw * 100 - implied)
            return _r(raw if as_percent <= as_fraction else raw * 100)
    return _r(raw)


def valuation(b) -> dict:
    q = b.quote
    return {
        "pe_ttm": _r(_num(q.get("trailingPE"))),
        "pe_forward": _r(_num(q.get("forwardPE"))),
        "pb": _r(_num(q.get("priceToBook"))),
        "ps_ttm": _r(_num(q.get("priceToSalesTrailing12Months"))),
        "ev_to_ebitda": _r(_num(q.get("enterpriseToEbitda"))),
        "ev_to_revenue": _r(_num(q.get("enterpriseToRevenue"))),
        "peg": _r(_num(q.get("trailingPegRatio"))),
        "dividend_yield_pct": _dividend_yield_pct(q),
        "market_cap": _num(q.get("marketCap")),
        "fifty_day_avg": _r(_num(q.get("fiftyDayAverage"))),
        "two_hundred_day_avg": _r(_num(q.get("twoHundredDayAverage"))),
    }


def technicals(b) -> dict:
    h = b.price_history
    if h is None or h.empty or "Close" not in h:
        return {k: None for k in (
            "last_close", "sma50", "sma200", "rsi14", "atr14_pct", "vol_ann_pct",
            "max_drawdown_pct", "high_52w", "low_52w", "pct_from_52w_high",
            "trend_vs_sma50", "trend_vs_sma200", "volume_vs_20d_avg",
        )}

    c = h["Close"].dropna()
    last = float(c.iloc[-1])
    sma50 = _r(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else None
    sma200 = _r(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None

    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = (up / dn).replace([math.inf, -math.inf], math.nan)
    rsi = 100 - 100 / (1 + rs)

    ret = c.pct_change().dropna()
    tr = (h["High"] - h["Low"]).dropna() if {"High", "Low"} <= set(h.columns) else None

    win = c.tail(252)
    hi = float(win.max()) if len(win) else None
    lo = float(win.min()) if len(win) else None
    dd = float(((c / c.cummax()) - 1).min() * 100)

    vol20 = _num(h["Volume"].tail(20).mean()) if "Volume" in h else None
    base = _num(h["Volume"].tail(60).mean()) if "Volume" in h else None

    return {
        "last_close": _r(last),
        "sma50": sma50,
        "sma200": sma200,
        "rsi14": _r(_num(rsi.iloc[-1]) if len(rsi.dropna()) else None),
        "atr14_pct": _r(_num((tr.tail(14).mean() / last * 100)) if tr is not None and len(tr) else None),
        "vol_ann_pct": _r(_num(ret.tail(252).std() * math.sqrt(252) * 100)),
        "max_drawdown_pct": _r(dd),
        "high_52w": _r(hi),
        "low_52w": _r(lo),
        "pct_from_52w_high": _pct(None if hi is None else last - hi, hi),
        "trend_vs_sma50": _pct(None if sma50 is None else last - sma50, sma50),
        "trend_vs_sma200": _pct(None if sma200 is None else last - sma200, sma200),
        "volume_vs_20d_avg": _pct(None if vol20 is None or base in (None, 0) else vol20 - base, base),
    }


def risk(b) -> dict:
    q, t = b.quote, technicals(b)
    ret_series = None
    if b.price_history is not None and not b.price_history.empty:
        ret_series = b.price_history["Close"].dropna().pct_change().dropna()
    return {
        "beta": _r(_num(q.get("beta"))),
        "annualised_vol_pct": t.get("vol_ann_pct"),
        "max_drawdown_pct": t.get("max_drawdown_pct"),
        "downside_deviation_pct": _r(
            _num(ret_series[ret_series < 0].tail(252).std() * math.sqrt(252) * 100)
            if ret_series is not None and len(ret_series[ret_series < 0]) > 5 else None
        ),
        "net_debt_to_equity": None,  # filled below by flows/governance if available
        "share_dilution_pct": None,
        "held_by_institutions_pct": _r(
            None if _num(q.get("heldPercentInstitutions")) is None else _num(q.get("heldPercentInstitutions")) * 100
        ),
        "short_ratio": _r(_num(q.get("shortRatio"))),
    }


def governance(b) -> dict:
    q = b.quote
    inc = b.income
    shares_now = _row_series(inc, "Diluted Average Shares", "Basic Average Shares")
    dilution = None
    if shares_now is not None and len(shares_now.dropna()) > 1:
        s = shares_now.dropna()
        dilution = _pct(float(s.iloc[0]) - float(s.iloc[-1]), float(s.iloc[-1]))

    return {
        "held_by_institutions_pct": _r(
            None if _num(q.get("heldPercentInstitutions")) is None else _num(q.get("heldPercentInstitutions")) * 100
        ),
        "held_by_insiders_pct": _r(
            None if _num(q.get("heldPercentInsiders")) is None else _num(q.get("heldPercentInsiders")) * 100
        ),
        "share_dilution_pct": dilution,
        "audit_risk": _num(q.get("auditRisk")),
        "overall_risk": _num(q.get("overallRisk")),
        "governance_risk": _num(q.get("governanceEpoch")),
        "compensation_risk": _num(q.get("compensationRisk")),
        "shares_outstanding": _num(q.get("sharesOutstanding")),
    }


PILLARS = ("fundamentals", "valuation", "technicals", "risk", "governance")

# Declared key lists, used by tests to catch metrics that are computed and cited
# by agents but missing from the rubric signal table (so they never affect score).
FUNDAMENTALS_KEYS = ("revenue", "revenue_growth_yoy_pct", "net_income", "gross_margin_pct",
                     "operating_margin_pct", "net_margin_pct", "fcf", "fcf_margin_pct",
                     "cash_conversion_ocf_over_ni", "debt_to_equity", "net_debt_to_ebitda",
                     "roa_pct", "roe_pct", "cash_to_assets_pct")
VALUATION_KEYS = ("pe_ttm", "pe_forward", "pb", "ps_ttm", "ev_to_ebitda", "ev_to_revenue",
                  "peg", "dividend_yield_pct", "market_cap", "fifty_day_avg", "two_hundred_day_avg")
TECHNICALS_KEYS = ("last_close", "sma50", "sma200", "rsi14", "atr14_pct", "vol_ann_pct",
                   "max_drawdown_pct", "high_52w", "low_52w", "pct_from_52w_high",
                   "trend_vs_sma50", "trend_vs_sma200", "volume_vs_20d_avg")
RISK_KEYS = ("beta", "annualised_vol_pct", "max_drawdown_pct", "downside_deviation_pct",
             "net_debt_to_equity", "share_dilution_pct", "held_by_institutions_pct", "short_ratio")
GOVERNANCE_KEYS = ("held_by_institutions_pct", "held_by_insiders_pct", "share_dilution_pct",
                   "audit_risk", "overall_risk", "governance_risk", "compensation_risk",
                   "shares_outstanding")
KEYS_BY_PILLAR = {
    "fundamentals": FUNDAMENTALS_KEYS, "valuation": VALUATION_KEYS,
    "technicals": TECHNICALS_KEYS, "risk": RISK_KEYS, "governance": GOVERNANCE_KEYS,
}


def all_pillars(b) -> dict:
    return {
        "fundamentals": fundamentals(b),
        "valuation": valuation(b),
        "technicals": technicals(b),
        "risk": risk(b),
        "governance": governance(b),
    }


def pillar_coverage(pillars: dict) -> dict:
    """Fraction of non-null fields per pillar. This is the article's 'evidence coverage'."""
    cov = {}
    for name, vals in pillars.items():
        total = len(vals)
        known = sum(1 for v in vals.values() if v is not None)
        cov[name] = round(known / total, 3) if total else 0.0
    return cov
