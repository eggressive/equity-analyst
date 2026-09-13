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


# Tokens that turn a label into a different line item rather than a qualified version
# of the same one. Both live wrong-row cases are covered by "non" and "other":
# "Net Non Operating Interest Income Expense" (BRK-B) and
# "Other Non Operating Income Expenses" (HDFCBANK.NS).
MEANING_CHANGERS = frozenset({"non", "other", "excluding", "except", "before", "prior"})


def _contains_run(haystack: list[str], needle: list[str]) -> bool:
    """True if needle appears as a contiguous run of tokens in haystack."""
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def _pick_label(df: pd.DataFrame, *needles: str) -> str | None:
    """Find the statement row that best matches a needle.

    Naive substring matching is unsafe here: the label
    "Other Non Operating Income Expenses" contains "operating income", and it
    sorts earlier in the index than the real "Operating Income" row. That single
    bug reported Reliance's operating margin as 0.34% instead of 11.48% and the
    Bear built an entire (wrong) thesis on it.

    Rules:
      1. Whole-word token matching, so "EBIT" never matches "EBITDA". The needle
         tokens must also appear as a *contiguous run*, so "Operating Income" cannot
         be assembled across an unrelated word ("Operating ... Income").
      2. An exact label match short-circuits everything.
      3. Extra tokens are allowed only when they qualify the same quantity. A
         candidate carrying a meaning changer ("non", "other", "excluding", ...) is a
         different line item and is refused. This is the rule that keeps a missing
         Operating Income row missing: BRK-B's nearest label is
         "Net Non Operating Interest Income Expense" and HDFCBANK.NS's is
         "Other Non Operating Income Expenses". Accepting either one published an
         operating margin of -1.23% / -1.91% and scored the pillar signal -2.
      4. Among accepted candidates: a leading match wins, then the fewest extra
         tokens, then the shorter label, then alphabetical order.
      5. If nothing qualifies, return None. Absence is never filled by the closest
         string: a missing row must leave the metric None so coverage can dilute.
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
            if not _contains_run(tk, nt):
                continue
            if (set(tk) - set(nt)) & MEANING_CHANGERS:
                continue
            leading = 0 if tk[:len(nt)] == nt else 1
            scored.append((leading, len(tk) - len(nt), len(tk), lbl))
        if scored:
            scored.sort()
            return scored[0][3]
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
    ocf = _row(cf, "Operating Cash Flow", "Total Cash From Operating Activities",
               "Cash Flow From Continuing Operating Activities")
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


# Share counts are read as a trend across the annual columns, not first against last, and the
# series is put back on one basis before it is measured. Three mechanisms replace the refusal;
# README failure mode 9 carries the measurements.
#
#   Repair     divide out an adjacent step that a recorded split factor explains, so the two
#              columns straddling a bonus land on one basis. The factor comes from the full
#              split history (`TickerBundle.splits`), never from the two-year price window: 9
#              of the 17 in-window splits in a 115-ticker sample are older than that window.
#              Only ratios at or below 0.9 or at or above 1.5 count: near-1 ratios are
#              distributions and ADR-ratio changes (SPGI 1.057, HON 0.9535, UL 0.888), and
#              those cannot rebase a share count.
#   Fallback   steps that stay implausible mean the columns are not on one basis, so the
#              restated quarterly series is measured instead, over its shorter window. The
#              same fallback covers a disagreement between the two series on one fiscal
#              period, because the quarterly series is the restated one.
#   Trend      the reported value is the change implied by the least-squares line through
#              log(average shares) across the window, so one jumped column cannot set it.
#
# Issuance stays scored. A count that doubles with no split in the window is an all-stock
# acquisition or equity financing: no factor explains it, so no mechanism touches it.
MAX_PLAUSIBLE_SHARE_RATIO = 2.0    # doubled in one step
MIN_PLAUSIBLE_SHARE_RATIO = 0.5    # halved in one step
ROUNDING_ALLOWANCE = 0.05          # HDB's ADS change reads 0.502 against a recorded 0.5
MATERIAL_SPLIT_RATIO = 1.5         # a 3:2 split or larger
REVERSE_SPLIT_RATIO = 0.9          # at or below this a recorded factor is a reverse split
SPLIT_MATCH_TOLERANCE = 0.10       # HDFCBANK's bonus step sits 7.75% above the recorded 2.0
BASIS_DISAGREEMENT_PCT = 5.0       # annual against quarterly for the same fiscal period
MIN_QUARTERS_FOR_TREND = 3         # two quarters are a comparison, not a trend


def _naive_ts(ts):
    """Statement columns are tz-naive, yfinance split dates are exchange-local."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _share_series(df):
    """Average share counts from one statement, newest column first, or None."""
    s = _row_series(df, "Diluted Average Shares", "Basic Average Shares")
    if s is None:
        return None
    s = s.dropna()
    return s if len(s) else None


def _recorded_split_factors(splits, columns) -> list:
    """Split and bonus factors dated inside the statement window, oldest first.

    Near-1 ratios are dropped: yfinance records spin-off distributions and ADR-ratio changes
    in the same series (SPGI 1.057, HON 0.9535, UL 0.888) and those cannot rebase a count.
    """
    if splits is None or len(splits) == 0 or columns is None or len(columns) == 0:
        return []
    dates = [_naive_ts(c) for c in columns]
    oldest, newest = min(dates), max(dates)
    out = []
    for date, ratio in splits.items():
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            continue
        if ratio <= 0 or not (ratio <= REVERSE_SPLIT_RATIO or ratio >= MATERIAL_SPLIT_RATIO):
            continue
        date = _naive_ts(date)
        if oldest <= date <= newest:
            out.append((date, ratio))
    return sorted(out)


def _step_matches_factor(step, factor) -> bool:
    """True when a step between two columns is the recorded corporate action.

    A restated column carries the action plus the movement of that period, in the direction
    issuance moves the count, so the step sits at the factor or up to SPLIT_MATCH_TOLERANCE
    above it whatever the direction of the action: HDFCBANK 2.155 = 2.0 x 1.078 for a bonus,
    and 0.105 = 0.1 x 1.05 for a reverse split with issuance on top. A step below the factor
    is not explained by it: only part of the count moved.
    """
    return 1 <= step / factor <= 1 + SPLIT_MATCH_TOLERANCE


def _repair_share_steps(series, factors):
    """Put the columns on one basis by dividing out the steps a split factor explains.

    A recorded corporate action happens once, so each factor repairs at most one step, and
    the step it repairs is the matching interval that holds the factor date or, when none
    does, the nearest matching interval. The nearest case is what a restated column looks
    like: HDFCBANK.NS's 2025-08-26 bonus appears as a step one column earlier, from
    2024-03-31 to 2025-03-31, because yfinance restates the newer column and not the older
    one. Bounding the factor to one step matters: three doublings against a single recorded
    2.0 would otherwise all be divided out and the dilution they represent would disappear.
    """
    dates = [_naive_ts(i) for i in series.index]
    values = [float(v) for v in series.values]
    order = sorted(range(len(dates)), key=lambda i: dates[i])  # oldest first
    steps = []
    for k in range(1, len(order)):
        previous, current = order[k - 1], order[k]
        if values[previous]:
            steps.append((k, dates[previous], dates[current], values[current] / values[previous]))

    scale = [1.0] * len(values)
    repairs = 0
    for factor_date, factor in factors:
        best = None
        for k, start, end, step in steps:
            if not _step_matches_factor(step, factor):
                continue
            if start < factor_date <= end:
                distance = 0
            else:
                distance = min(abs((factor_date - end).days), abs((start - factor_date).days))
            if best is None or distance < best[0]:
                best = (distance, k)
        if best is None:
            continue
        for i in order[:best[1]]:
            scale[i] *= factor
        repairs += 1
    return pd.Series([v * s for v, s in zip(values, scale)], index=series.index), repairs


def _implausible_share_steps(series):
    """Steps that no single year of issuance explains, split into directions.

    Returns (up, down), in which either list non-empty means the count moved further than
    issuance can in one year. One direction alone can be real: two doublings in a row are
    issuance. A step up and a step down in the same window cannot be one corporate action, so
    that pair is what marks mixed bases.

    ROUNDING_ALLOWANCE is there because recorded ratios are rounded: HDB's ADS change reads
    0.502 on the share series, and 0.4% must not decide whether the columns mix bases.
    """
    dates = [_naive_ts(i) for i in series.index]
    values = [float(v) for v in series.values]
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    upper = MAX_PLAUSIBLE_SHARE_RATIO / (1 + ROUNDING_ALLOWANCE)
    lower = MIN_PLAUSIBLE_SHARE_RATIO * (1 + ROUNDING_ALLOWANCE)
    up, down = [], []
    for k in range(1, len(order)):
        previous, current = order[k - 1], order[k]
        if not values[previous]:
            continue
        step = values[current] / values[previous]
        if step >= upper:
            up.append(round(step, 3))
        elif step <= lower:
            down.append(round(step, 3))
    return up, down


def _share_trend_pct(series):
    """Change implied by the least-squares line through log(average shares), in percent.

    First column against last gives the same number for a series that grows steadily, and it
    is damped when one column jumps, which is the point of reading a trend instead of two
    endpoints.
    """
    if series is None or len(series) < 2:
        return None
    dates = [_naive_ts(i) for i in series.index]
    values = [float(v) for v in series.values]
    if any(v <= 0 for v in values) or len(set(dates)) < 2:
        return None
    t = [(d - min(dates)).days / 365.25 for d in dates]
    y = [math.log(v) for v in values]
    mean_t, mean_y = sum(t) / len(t), sum(y) / len(y)
    spread = sum((x - mean_t) ** 2 for x in t)
    if spread == 0:
        return None
    slope = sum((x - mean_t) * (v - mean_y) for x, v in zip(t, y)) / spread
    return round((math.exp(slope * (max(t) - min(t))) - 1) * 100, 2)


def _basis_disagreement_pct(annual, quarterly):
    """Worst gap between the two share series for one fiscal period, in percent.

    The quarterly series is the restated one, so a gap means the annual columns carry a basis
    the quarterly series does not.
    """
    if annual is None or quarterly is None:
        return None
    restated = {_naive_ts(i): float(v) for i, v in quarterly.items()}
    worst = None
    for i, v in annual.items():
        date = _naive_ts(i)
        if restated.get(date):
            gap = abs(float(v) / restated[date] - 1) * 100
            worst = gap if worst is None else max(worst, gap)
    return None if worst is None else round(worst, 2)


def governance(b) -> dict:
    """Ownership metrics, including the trend in average shares.

    `share_dilution_pct` is the change implied by the least-squares line through log(average
    shares) across the columns yfinance returns, after the series is put on one basis. The
    annual columns are used first, repaired against the full split history. When steps stay
    implausible, or when the annual and quarterly series disagree on one fiscal period, the
    restated quarterly series is measured instead. A refusal survives for the case where
    neither series is usable, and it is ordinary missing data: governance coverage drops and
    the dilution signal leaves the pillar rather than scoring a corporate action as dilution.
    """
    q = b.quote
    annual = _share_series(b.income)
    quarterly = _share_series(getattr(b, "quarterly_income", None))
    if annual is not None and len(annual) < 2:
        annual = None
    if quarterly is not None and len(quarterly) < MIN_QUARTERS_FOR_TREND:
        quarterly = None

    dilution = None
    if annual is not None or quarterly is not None:
        series = annual if annual is not None else quarterly
        if annual is not None:
            factors = _recorded_split_factors(getattr(b, "splits", None), annual.index)
            repaired, _ = _repair_share_steps(annual, factors)
            up, down = _implausible_share_steps(repaired)
            gap = _basis_disagreement_pct(annual, quarterly)
            mixed_bases = (
                bool(up) and bool(down)          # two bases, not one corporate action
                or (len(up) + len(down) == 1 and bool(factors))  # ambiguous beside a split
                or (gap is not None and gap > BASIS_DISAGREEMENT_PCT)
            )
            series = quarterly if mixed_bases else repaired
        dilution = _share_trend_pct(series)

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
