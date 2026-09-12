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
