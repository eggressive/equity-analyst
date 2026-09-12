"""Evidence layer. All raw numbers. No LLM anywhere in this file.

Sources (all free, no API key required):
  - yfinance: prices, OHLCV history, financial statements, quote metadata. US + NSE (.NS).
  - SEC EDGAR XBRL companyfacts: the fetch and ticker-to-CIK helpers exist below, but
    nothing in the pipeline calls them yet. Every number in a run comes from yfinance.

Provenance: sourced scalar fields (`Evidence`) carry a source string. The metric
values inside the pillars are keyed by pillar and attributed to the data layer as a
whole; they do not carry a per-value source string.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

# SEC EDGAR returns 403 for a User-Agent whose contact is not a plausible email
# (dimitar@localhost is rejected, dimitar@example.com is accepted). Keep a real
# domain in the contact string or every SEC call silently fails.
SEC_UA = "EquityAnalyst-Research/0.1 (personal research; dimitar@example.com)"


def _num(x):
    try:
        f = float(x)
        return None if (f != f) else f  # NaN guard without importing numpy
    except (TypeError, ValueError):
        return None


@dataclass
class Evidence:
    """A single sourced fact. The unit of auditability."""

    key: str
    value: object
    source: str
    as_of: str
    coverage: bool = True

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "as_of": self.as_of,
        }


@dataclass
class TickerBundle:
    symbol: str
    market: str
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    income: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_income: pd.DataFrame = field(default_factory=pd.DataFrame)
    quote: dict = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def evidence_coverage(self) -> float:
        """0..1 fraction of the 4 statement blocks that actually loaded.

        This is the gate the article describes: a pillar backed by strong evidence
        must not be silently weighted the same as one working on thin data.
        """
        blocks = [self.price_history, self.income, self.balance, self.cashflow]
        return sum(1 for b in blocks if not b.empty) / len(blocks)

    def add(self, key: str, value: object, source: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.evidence.append(Evidence(key, value, source, ts, coverage=value is not None))
        return None


def detect_market(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return "NSE" if s.endswith(".NS") else "BSE"
    return "US"


def load(symbol: str, history_period: str = "2y") -> TickerBundle:
    """Pull everything for one ticker. Never raises on missing blocks."""
    t = yf.Ticker(symbol)
    b = TickerBundle(symbol=symbol.upper(), market=detect_market(symbol))

    try:
        b.price_history = t.history(period=history_period, auto_adjust=True)
        if b.price_history.empty:
            b.warnings.append("no price history returned")
    except Exception as e:  # network / delisting / bad symbol
        b.warnings.append(f"price history failed: {type(e).__name__}")

    # yfinance 1.7 exposes statements as properties, not callables.
    for attr, target in (
        ("income_stmt", "income"),
        ("balance_sheet", "balance"),
        ("cash_flow", "cashflow"),
        ("quarterly_income_stmt", "quarterly_income"),
    ):
        try:
            df = getattr(t, attr)
            setattr(b, target, df if df is not None else pd.DataFrame())
        except Exception as e:
            b.warnings.append(f"{attr} failed: {type(e).__name__}")

    try:
        b.quote = t.info or {}
        if not b.quote:
            b.warnings.append("quote metadata empty (rate limited or unsupported market)")
    except Exception as e:
        b.warnings.append(f"quote metadata failed: {type(e).__name__}")

    b.add("last_price", _last(b.price_history), "yfinance:history")
    b.add("market_cap", b.quote.get("marketCap"), "yfinance:info")
    b.add("sector", b.quote.get("sector"), "yfinance:info")
    return b


def load_extras(symbol: str) -> dict:
    """Sentiment / events / flow inputs. Separate call so a failure here
    degrades those three pillars to low coverage instead of killing the run."""
    t = yf.Ticker(symbol)
    out: dict = {"news": [], "calendar": {}, "holders": {}, "warnings": []}

    try:
        news = t.news or []
        out["news"] = [
            {
                "title": n.get("title"),
                "publisher": (n.get("publisher") or ""),
                "published": n.get("providerPublishTime"),
                "link": n.get("link"),
            }
            for n in news[:15]
        ]
    except Exception as e:
        out["warnings"].append(f"news failed: {type(e).__name__}")

    try:
        cal = t.calendar or {}
        out["calendar"] = {
            k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in (cal.items() if isinstance(cal, dict) else [])
        }
    except Exception as e:
        out["warnings"].append(f"calendar failed: {type(e).__name__}")

    try:
        inst = t.institutional_holders
        if inst is not None and not inst.empty:
            out["holders"]["institutional"] = [
                {"holder": str(r.get("Holder", ""))[:60], "pct": _num(r.get("% Out")), "value": _num(r.get("Value"))}
                for _, r in inst.head(10).iterrows()
            ]
    except Exception as e:
        out["warnings"].append(f"holders failed: {type(e).__name__}")

    return out


def _last(df: pd.DataFrame, col: str = "Close"):
    if df is None or df.empty or col not in df.columns:
        return None
    v = df[col].dropna()
    return float(v.iloc[-1]) if len(v) else None


def sec_companyfacts(cik: str | int) -> dict:
    """Authoritative US filings data. cik is the integer, not zero-padded."""
    cik_int = int(str(cik).lstrip("0") or 0)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_int:010d}.json"
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def resolve_cik(ticker: str) -> str | None:
    """Map a US ticker to its SEC CIK using the official ticker file."""
    req = urllib.request.Request(
        "https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": SEC_UA}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    want = ticker.upper()
    for row in data.values():
        if row.get("ticker", "").upper() == want:
            return str(row["cik_str"])
    return None
