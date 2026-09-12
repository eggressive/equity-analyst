"""Statement-row matching tests.

These guard the most damaging class of bug in a financial pipeline: silently
picking the wrong line item and reporting a confidently wrong number.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import metrics  # noqa: E402

# Mirrors the real yfinance row set that caused the Reliance bug.
INCOME = pd.DataFrame(
    {"2026": [10572190000000.0, 7868240000000.0, 2703950000000.0, 1490180000000.0,
              1213770000000.0, 36140000000.0, 778460000000.0, 807750000000.0, 1213770000000.0]},
    index=["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Expense",
           "Operating Income", "Other Non Operating Income Expenses",
           "Other Operating Expenses", "Net Income", "EBIT"],
)


class FakeBundle:
    def __init__(self, income):
        self.income = income
        self.balance = pd.DataFrame()
        self.cashflow = pd.DataFrame()
        self.quote = {}
        self.price_history = pd.DataFrame()


def test_operating_income_picks_exact_row_not_substring():
    """'Other Non Operating Income Expenses' must never satisfy 'Operating Income'."""
    assert metrics._pick_label(INCOME, "Operating Income") == "Operating Income"


def test_operating_margin_is_not_zero():
    f = metrics.fundamentals(FakeBundle(INCOME))
    assert f["revenue"] == 10572190000000.0
    assert abs(f["operating_margin_pct"] - 11.48) < 0.05, f["operating_margin_pct"]


def test_ebit_does_not_match_ebitda():
    df = pd.DataFrame({"2026": [100.0, 200.0]}, index=["EBIT", "EBITDA"])
    assert metrics._pick_label(df, "EBIT") == "EBIT"


def test_fewest_extra_tokens_wins():
    df = pd.DataFrame({"2026": [1.0, 2.0, 3.0]},
                      index=["Operating Income", "Operating Income As Reported",
                             "Other Non Operating Income Expenses"])
    # exact match wins
    assert metrics._pick_label(df, "Operating Income") == "Operating Income"
    # a needle with no exact match prefers the tightest candidate
    assert metrics._pick_label(df, "Income As Reported") == "Operating Income As Reported"


def test_missing_needle_returns_none_not_a_wrong_row():
    assert metrics._pick_label(INCOME, "Nonexistent Line Item") is None
    assert metrics._row(INCOME, "Nonexistent Line Item") is None


def test_total_debt_fallback_uses_balance_sheet_only():
    p = metrics.fundamentals(FakeBundle(INCOME))
    assert p["debt_to_equity"] is None  # no balance sheet in the fake bundle


def test_dividend_yield_is_not_scaled_twice():
    """yfinance 1.7 returns dividendYield already as a percent. AAPL reports 0.33 for a
    ~0.33% yield; multiplying by 100 published 33.0% as a real metric, and every agent in
    the run spent a line treating it as a data defect."""
    b = FakeBundle(INCOME)
    b.quote = {"dividendYield": 0.33, "dividendRate": 1.08, "regularMarketPrice": 332.27}
    assert metrics.valuation(b)["dividend_yield_pct"] == 0.33


def test_dividend_yield_fraction_style_provider_is_still_rescaled():
    """Older yfinance builds returned a fraction. The cross-check against
    dividendRate/price must keep working for that shape too."""
    b = FakeBundle(INCOME)
    b.quote = {"dividendYield": 0.0033, "dividendRate": 1.08, "regularMarketPrice": 332.27}
    assert metrics.valuation(b)["dividend_yield_pct"] == 0.33


def test_dividend_yield_without_cross_check_uses_percent_semantics():
    b = FakeBundle(INCOME)
    b.quote = {"dividendYield": 2.4}
    assert metrics.valuation(b)["dividend_yield_pct"] == 2.4


def test_missing_debt_rows_do_not_score_as_debt_free():
    """Absence must never score. `or 0` used to turn a missing debt row into a real zero,
    and the leverage table maps 0.0 to +2, the most bullish value it can emit."""
    b = FakeBundle(INCOME)
    b.balance = pd.DataFrame({"2026": [500.0, 60.0]},
                             index=["Total Assets", "Stockholders Equity"])
    assert metrics.fundamentals(b)["debt_to_equity"] is None


def test_debt_fallback_keeps_short_term_debt():
    """The combined capital-lease row excludes current debt and used to overwrite the
    long+short sum, understating leverage on any balance sheet without a Total Debt row."""
    b = FakeBundle(INCOME)
    b.balance = pd.DataFrame(
        {"2026": [600.0, 100.0, 400.0]},
        index=["Long Term Debt And Capital Lease Obligation", "Current Debt",
               "Stockholders Equity"],
    )
    assert metrics.fundamentals(b)["debt_to_equity"] == 1.75  # (600 + 100) / 400


if __name__ == "__main__":
    import traceback

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"ERROR {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    raise SystemExit(1 if failed else 0)
