"""Statement-row matching tests.

These guard the most damaging class of bug in a financial pipeline: silently
picking the wrong line item and reporting a confidently wrong number.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import metrics  # noqa: E402
import rubric  # noqa: E402

# Mirrors the real yfinance row set that caused the Reliance bug.
INCOME = pd.DataFrame(
    {"2026": [10572190000000.0, 7868240000000.0, 2703950000000.0, 1490180000000.0,
              1213770000000.0, 36140000000.0, 778460000000.0, 807750000000.0, 1213770000000.0]},
    index=["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Expense",
           "Operating Income", "Other Non Operating Income Expenses",
           "Other Operating Expenses", "Net Income", "EBIT"],
)


class FakeBundle:
    def __init__(self, income, splits=None, quarterly_income=None):
        self.income = income
        self.quarterly_income = pd.DataFrame() if quarterly_income is None else quarterly_income
        self.balance = pd.DataFrame()
        self.cashflow = pd.DataFrame()
        self.quote = {}
        self.price_history = pd.DataFrame()
        self.splits = pd.Series(dtype=float) if splits is None else splits


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


# The real yfinance label sets that exposed this. Neither has an Operating Income row,
# and in both cases the nearest label is a different line item.
BRK_INCOME = pd.DataFrame(
    {"2026": [410522000000.0, -5069000000.0, -4000.0],
     "2025": [364482000000.0, -4500000000.0, -3800.0]},
    index=["Total Revenue", "Net Non Operating Interest Income Expense",
           "Interest Expense Non Operating"],
)
HDFC_INCOME = pd.DataFrame(
    {"2026": [1925667800000.0, -36789600000.0],
     "2025": [1800000000000.0, -30000000000.0]},
    index=["Operating Revenue", "Other Non Operating Income Expenses"],
)


def test_negated_label_is_not_operating_income():
    """BRK-B has no Operating Income row. "Net Non Operating Interest Income Expense"
    contains the tokens but is a different line item; it used to be returned and scored
    the operating_margin signal -2 against a fabricated -1.23%."""
    assert metrics._pick_label(BRK_INCOME, "Operating Income", "EBIT") is None
    assert metrics.fundamentals(FakeBundle(BRK_INCOME))["operating_margin_pct"] is None


def test_other_non_operating_row_is_not_operating_income():
    """HDFCBANK.NS has no Operating Income row either. "Other Non Operating Income
    Expenses" used to be picked, for a fabricated margin of -1.91%."""
    assert metrics._pick_label(HDFC_INCOME, "Operating Income", "EBIT") is None
    assert metrics.fundamentals(FakeBundle(HDFC_INCOME))["operating_margin_pct"] is None


def test_refused_row_reaches_the_rubric_as_unavailable_not_as_a_number():
    """The rubric's rule is that absence dilutes coverage instead of being scored. A row
    we refuse to pick must therefore arrive at the signal table as None."""
    f = metrics.fundamentals(FakeBundle(BRK_INCOME))
    assert rubric.build_signals({"fundamentals": f})["fundamentals"]["operating_margin"] is None


def test_qualifying_suffix_is_still_accepted():
    """Extra tokens are only a problem when they change the meaning. "As Reported"
    qualifies the same quantity, so it must still match."""
    df = pd.DataFrame({"2026": [1.0]}, index=["Operating Income As Reported"])
    assert metrics._pick_label(df, "Operating Income") == "Operating Income As Reported"


def test_needle_tokens_must_appear_as_a_contiguous_run():
    """A needle must not be assembled across an unrelated word: "Operating Lease Income"
    is not "Operating Income"."""
    df = pd.DataFrame({"2026": [1.0]}, index=["Operating Lease Income"])
    assert metrics._pick_label(df, "Operating Income") is None


def test_absent_operating_income_falls_through_to_ebit():
    """Live BRK-B has no Operating Income row but does carry EBIT, so the second needle
    is what rescues the metric: -1.23% from the interest row became 21.32% = EBIT /
    revenue. The fixture above models the tickers where EBIT is absent too."""
    df = pd.DataFrame({"2026": [410522000000.0, 87528000000.0]},
                      index=["Total Revenue", "EBIT"])
    assert metrics._pick_label(df, "Operating Income", "EBIT") == "EBIT"
    assert metrics.fundamentals(FakeBundle(df))["operating_margin_pct"] == 21.32


ASML_CASHFLOW = pd.DataFrame(
    {"2026": [12658500000.0, -1631200000.0, 11027300000.0]},
    index=["Cash Flow From Continuing Operating Activities", "Capital Expenditure",
           "Free Cash Flow"],
)


def test_continued_operations_cash_flow_is_operating_cash_flow():
    """ASML labels its operating cash flow row "Cash Flow From Continuing Operating
    Activities". That phrase does not contain "Operating Cash Flow" contiguously, so
    without its own needle the cash conversion metric disappears (1.32 -> None) and one
    rubric signal drops out with it."""
    inc = pd.DataFrame({"2026": [28000000000.0, 9600000000.0]},
                       index=["Total Revenue", "Net Income"])
    b = FakeBundle(inc)
    b.cashflow = ASML_CASHFLOW
    f = metrics.fundamentals(b)
    assert f["cash_conversion_ocf_over_ni"] == 1.32
    # and the row really is operating cash flow: OCF + capex equals the statement's FCF
    assert f["fcf"] == 11027300000.0


# The live HDFCBANK.NS annual columns, exact: FY2026 and FY2025 are restated to the
# post-bonus basis, FY2024 and FY2023 are not.
HDFC_ANNUAL = pd.DataFrame(
    {"2026-03-31": [15406371571.0], "2025-03-31": [15318868212.0],
     "2024-03-31": [7107452428.0], "2023-03-31": [5587173328.0]},
    index=["Diluted Average Shares"],
)
HDFC_SPLITS = pd.Series([5.0, 2.0, 2.0],
                        index=pd.to_datetime(["2011-07-14", "2019-09-19", "2025-08-26"]))


def test_bonus_step_is_repaired_instead_of_refused():
    """HDFCBANK.NS: yfinance restates the annual columns after the August 2025 1:1 bonus for
    FY2025 and FY2026 but not for FY2024 and FY2023, so first against last reads 7.107bn ->
    15.406bn = +175.75% and a corporate action scores as dilution. The observed step is 2.155
    against a recorded 2.0, and the extra 7.8% is that year's real issuance, so the two
    pre-bonus columns are rebased and the metric reports what is left: the merger, +36.56%
    across the window, which keeps its -2."""
    gov = metrics.governance(FakeBundle(HDFC_ANNUAL, splits=HDFC_SPLITS))
    assert gov["share_dilution_pct"] == 36.56
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] == -2


TRV_ANNUAL = pd.DataFrame(
    {"2025-12-31": [227600000.0], "2024-12-31": [23110000000.0],
     "2023-12-31": [232200000.0], "2022-12-31": [239700000.0]},
    index=["Diluted Average Shares"],
)
TRV_QUARTERLY = pd.DataFrame(
    {"2026-06-30": [213600000.0], "2026-03-31": [218400000.0], "2025-12-31": [224000000.0],
     "2025-09-30": [227500000.0], "2025-06-30": [229300000.0]},
    index=["Diluted Average Shares"],
)


def test_columns_on_different_bases_fall_back_to_the_restated_quarterly_series():
    """TRV: the 2024 annual column is 23.11bn where the years either side are 0.23bn, so one
    step reads as 99x and the next as 0.01x. No split explains either step and no single year
    of issuance produces them, so the annual columns are not one series. The restated
    quarterly columns are measured instead: -7.04% across their shorter window, a buyback."""
    b = FakeBundle(TRV_ANNUAL, quarterly_income=TRV_QUARTERLY)
    gov = metrics.governance(b)
    assert gov["share_dilution_pct"] == -7.04
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] == 2


HDB_ANNUAL = pd.DataFrame(
    {"2025-03-31": [5106000000.0], "2024-03-31": [4738000000.0],
     "2023-03-31": [1862000000.0], "2022-03-31": [3709000000.0]},
    index=["Diluted Average Shares"],
)
HDB_QUARTERLY = pd.DataFrame({"2025-06-30": [2566000000.0], "2024-12-31": [2559000000.0]},
                             index=["Diluted Average Shares"])
HDB_SPLITS = pd.Series([5.0, 2.0, 2.0],
                       index=pd.to_datetime(["2011-07-25", "2019-09-26", "2025-09-08"]))


def test_refusal_survives_when_neither_series_is_usable():
    """HDB, an ADS listing: yfinance mixes per-ADS and per-share counts across the annual
    columns (3.709bn -> 1.862bn -> 4.738bn), and the 2025-09-08 factor of 2.0 sits after the
    newest column, so no recorded action explains either step. The quarterly series has two
    columns, which is a comparison rather than a trend. Refuse, and let coverage dilute."""
    b = FakeBundle(HDB_ANNUAL, splits=HDB_SPLITS, quarterly_income=HDB_QUARTERLY)
    gov = metrics.governance(b)
    assert gov["share_dilution_pct"] is None
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] is None


XOM_ANNUAL = pd.DataFrame(
    {"2025-12-31": [4305100000.0], "2024-12-31": [4298000000.0],
     "2023-12-31": [4052000000.0], "2022-12-31": [4205000000.0]},
    index=["Diluted Average Shares"],
)
AAPL_ANNUAL = pd.DataFrame(
    {"2025-09-30": [15005000000.0], "2024-09-30": [15408000000.0],
     "2023-09-30": [15813000000.0], "2022-09-30": [16326000000.0]},
    index=["Diluted Average Shares"],
)


def test_trend_reading_survives_one_jumped_column():
    """XOM: first against last reads +2.38% because the 2022 column is high, while the line
    through the four columns says +3.96%: the 2024 issue is what the endpoints hide. The
    trend is the reading that catches it, and it crosses the -1 band into -2."""
    gov = metrics.governance(FakeBundle(XOM_ANNUAL))
    assert gov["share_dilution_pct"] == 3.96
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] == -2


def test_trend_equals_the_endpoint_change_for_a_steady_series():
    """AAPL buys back steadily, so the fitted trend and first against last agree to within
    0.1pp: -8.03% against -8.09%. A definition change that moves this case would be a bug."""
    gov = metrics.governance(FakeBundle(AAPL_ANNUAL))
    assert gov["share_dilution_pct"] == -8.03
    assert abs(gov["share_dilution_pct"] - (-8.09)) < 0.1
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] == 2


BASIS_ANNUAL = pd.DataFrame(
    {"2025-03-31": [1000000000.0], "2024-03-31": [1000000000.0], "2023-03-31": [1000000000.0]},
    index=["Diluted Average Shares"],
)
BASIS_QUARTERLY = pd.DataFrame(
    {"2025-03-31": [1120000000.0], "2024-12-31": [1100000000.0], "2024-09-30": [1080000000.0]},
    index=["Diluted Average Shares"],
)


def test_annual_and_quarterly_disagreement_prefers_the_quarterly_series():
    """The basis check: the annual column for a fiscal period is 12% below the quarterly
    column for the same period, so the annual series carries a basis the restated quarterly
    series does not. The quarterly series is what gets measured."""
    b = FakeBundle(BASIS_ANNUAL, quarterly_income=BASIS_QUARTERLY)
    gov = metrics.governance(b)
    assert gov["share_dilution_pct"] == 3.7


ONE_SPLIT_THREE_DOUBLINGS = pd.DataFrame(
    {"2025-03-31": [800.0], "2024-03-31": [400.0], "2023-03-31": [200.0],
     "2022-03-31": [100.0]},
    index=["Diluted Average Shares"],
)
LAST_INTERVAL_SPLIT = pd.Series([2.0], index=pd.to_datetime(["2025-01-15"]))


def test_one_factor_repairs_one_step_only():
    """Three doublings with a single recorded 2:1 split dated in the last interval. Matching
    the factor against every step divides out all three and reports 0.0%, erasing two real
    doublings. A corporate action happens once, so the factor repairs the one interval that
    holds its date: the series reads +328.64% and the two doublings keep their -2."""
    gov = metrics.governance(FakeBundle(ONE_SPLIT_THREE_DOUBLINGS, splits=LAST_INTERVAL_SPLIT))
    assert gov["share_dilution_pct"] == 328.64
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] == -2


REVERSE_WITH_ISSUANCE = pd.DataFrame(
    {"2025-03-31": [105000000.0], "2024-03-31": [1000000000.0]},
    index=["Diluted Average Shares"],
)
REVERSE_FACTOR = pd.Series([0.1], index=pd.to_datetime(["2024-06-01"]))


def test_reverse_split_with_issuance_is_repaired():
    """A recorded 0.1 reverse split with issuance on top reads 0.105, which is the factor
    times 1.05, and it must match on the same terms as a forward split: the step is the action
    plus the movement of that period. Rejecting it would treat one step as mixed bases and drop
    the metric to the quarterly series or to None."""
    gov = metrics.governance(FakeBundle(REVERSE_WITH_ISSUANCE, splits=REVERSE_FACTOR))
    assert gov["share_dilution_pct"] == 5.0


def test_real_issuance_inside_the_band_keeps_its_number():
    """Realty Income issues equity repeatedly: +48.5% over the window is issuance, not
    an artefact, and must still be reported and scored."""
    df = pd.DataFrame({"2026": [911000000.0], "2022": [613000000.0]},
                      index=["Diluted Average Shares"])
    gov = metrics.governance(FakeBundle(df))
    assert gov["share_dilution_pct"] == 48.61
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] == -2


def test_real_buyback_inside_the_band_keeps_its_number():
    """AIG bought back steadily: -27.62% is real and scores +2."""
    df = pd.DataFrame({"2025": [570000000.0], "2022": [788000000.0]},
                      index=["Diluted Average Shares"])
    gov = metrics.governance(FakeBundle(df))
    assert gov["share_dilution_pct"] == -27.66
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] == 2


DOUBLED = pd.DataFrame({"2026": [2000000000.0], "2022": [1000000000.0]},
                       index=["Diluted Average Shares"])
JUST_UNDER_DOUBLE = pd.DataFrame({"2026": [1990000000.0], "2022": [1000000000.0]},
                                 index=["Diluted Average Shares"])
HALVED = pd.DataFrame({"2026": [500000000.0], "2022": [1000000000.0]},
                      index=["Diluted Average Shares"])
INSIDE_WINDOW = pd.Series([2.0], index=pd.to_datetime(["2024-06-01"]))
OUTSIDE_WINDOW = pd.Series([2.0], index=pd.to_datetime(["2019-06-01"]))
SMALL_ADJUSTMENT = pd.Series([1.06], index=pd.to_datetime(["2024-06-01"]))


def test_doubling_the_recorded_factor_explains_is_repaired_to_the_factor():
    """A step that is exactly the recorded 2.0 is the split, so the older column is rebased
    and what remains is nothing: 0.0%, not a refusal and not 100% of dilution."""
    gov = metrics.governance(FakeBundle(DOUBLED, splits=INSIDE_WINDOW))
    assert gov["share_dilution_pct"] == 0.0
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] == 0


def test_step_below_the_recorded_factor_is_ambiguous_not_issuance():
    """1.99 against a recorded 2.0 is not the split plus issuance, and it is not a doubling
    either: only part of the count moved. Beside a recorded split the honest answer is a
    refusal, so coverage dilutes instead of a confident 99%."""
    assert metrics.governance(
        FakeBundle(JUST_UNDER_DOUBLE, splits=INSIDE_WINDOW))["share_dilution_pct"] is None
    assert metrics.governance(
        FakeBundle(HALVED, splits=INSIDE_WINDOW))["share_dilution_pct"] is None


def test_doubling_without_a_split_keeps_the_issuance_score():
    """A doubling with no split in the statement window is issuance: an all-stock
    acquisition or sustained equity financing. Refusing it would delete the -2 dilution
    score and flatter the company, so the number must be kept."""
    gov = metrics.governance(FakeBundle(DOUBLED))
    assert gov["share_dilution_pct"] == 100.0
    assert rubric.build_signals({"governance": gov})["governance"]["dilution"] == -2


def test_reverse_split_shaped_drop_is_refused():
    """A count that halves or falls further over the window is the same artefact in the
    other direction (a reverse split restates the newer columns upward)."""
    df = pd.DataFrame({"2026": [1000000000.0], "2022": [20000000000.0]},
                      index=["Diluted Average Shares"])
    reverse = pd.Series([0.1], index=pd.to_datetime(["2025-01-15"]))
    assert metrics.governance(FakeBundle(df, splits=reverse))["share_dilution_pct"] is None


def test_split_outside_the_statement_window_is_not_evidence():
    """A split older than the oldest annual column cannot rebase the series, so it must
    not trigger a refusal."""
    gov = metrics.governance(FakeBundle(DOUBLED, splits=OUTSIDE_WINDOW))
    assert gov["share_dilution_pct"] == 100.0


def test_small_split_adjustment_is_not_evidence():
    """yfinance records spin-off adjustments as splits too, for example Honeywell at
    1.061 and 0.9535. Those are far too small to rebase a share series, so they must not
    let a real doubling escape its score."""
    gov = metrics.governance(FakeBundle(DOUBLED, splits=SMALL_ADJUSTMENT))
    assert gov["share_dilution_pct"] == 100.0


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
