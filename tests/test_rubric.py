"""Determinism and invariant tests for the scoring layers.

The design claim is: given the same numbers, the verdict is always identical, and
missing data can never be laundered into a neutral score.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agents  # noqa: E402
import rubric  # noqa: E402
import verify  # noqa: E402

PILLARS = {
    "fundamentals": {
        "revenue_growth_yoy_pct": 18.0, "net_margin_pct": 22.0, "fcf_margin_pct": 25.0,
        "cash_conversion_ocf_over_ni": 1.2, "debt_to_equity": 0.25, "roe_pct": 30.0,
    },
    "valuation": {"pe_ttm": 18.0, "pe_forward": 16.0, "pb": 3.0, "ev_to_ebitda": 11.0, "ps_ttm": 3.0},
    "technicals": {
        "trend_vs_sma200": 12.0, "trend_vs_sma50": 5.0, "rsi14": 58.0, "pct_from_52w_high": -3.0,
    },
    "risk": {"annualised_vol_pct": 26.0, "max_drawdown_pct": -22.0, "beta": 0.95},
    "governance": {
        "held_by_institutions_pct": 62.0, "held_by_insiders_pct": 4.0, "share_dilution_pct": -1.5,
    },
}


def _score(pillars, horizon):
    return rubric.score(rubric.build_signals(pillars), horizon)


def test_same_input_same_verdict():
    a = _score(PILLARS, "LONG_TERM")
    b = _score(PILLARS, "LONG_TERM")
    assert a == b, "rubric is not deterministic"


def test_strong_company_scores_bullish_long():
    r = _score(PILLARS, "LONG_TERM")
    assert r["verdict"] in ("BULLISH", "STRONG_BULLISH"), r
    assert r["confidence"] > 0.9


def test_short_term_ignores_long_term_strength():
    """A great business with weak price data must not get a short-term buy."""
    p = {k: dict(v) for k, v in PILLARS.items()}
    p["technicals"] = {"trend_vs_sma200": -14.0, "trend_vs_sma50": -9.0, "rsi14": 34.0,
                       "pct_from_52w_high": -28.0}
    p["risk"] = {"annualised_vol_pct": 55.0, "max_drawdown_pct": -50.0, "beta": 1.6}
    short = _score(p, "SHORT_TERM")
    long_ = _score(p, "LONG_TERM")
    assert short["overall_score"] < long_["overall_score"]
    assert short["verdict"] in ("BEARISH", "STRONG_BEARISH", "NEUTRAL"), short


def test_missing_metrics_reduce_confidence_not_score_to_zero():
    thin = {k: {} for k in PILLARS}
    r = _score(thin, "LONG_TERM")
    assert r["verdict"] == "INSUFFICIENT_DATA", r
    assert r["confidence"] < rubric.COVERAGE_FLOOR


def test_partial_coverage_dilutes_weight():
    full = _score(PILLARS, "LONG_TERM")
    partial = {k: dict(v) for k, v in PILLARS.items()}
    partial["fundamentals"] = {"revenue_growth_yoy_pct": 18.0}
    p = _score(partial, "LONG_TERM")
    assert p["confidence"] < full["confidence"]
    assert p["pillars"]["fundamentals"]["coverage"] < 1.0


def test_low_leverage_scores_positive_and_high_scores_negative():
    """Regression guard: an inverted signal table once made low-debt companies
    read as highly leveraged. Direction must be a property of the table."""
    low = rubric.build_signals({"fundamentals": {"debt_to_equity": 0.1},
                                "valuation": {"pe_ttm": 8.0},
                                "risk": {"beta": 0.6, "annualised_vol_pct": 15.0}})
    high = rubric.build_signals({"fundamentals": {"debt_to_equity": 3.0},
                                 "valuation": {"pe_ttm": 90.0},
                                 "risk": {"beta": 2.4, "annualised_vol_pct": 75.0}})
    assert low["fundamentals"]["leverage"] > 0 > high["fundamentals"]["leverage"]
    assert low["valuation"]["pe_ttm"] > 0 > high["valuation"]["pe_ttm"]
    assert low["risk"]["beta"] > 0 > high["risk"]["beta"]
    assert low["risk"]["volatility"] > 0 > high["risk"]["volatility"]


def test_verification_flags_fabricated_number():
    text = "Revenue grew 18.0% and net margin was 22.0%. The stock trades at 999.5x earnings."
    v = verify.verify_claims(text, PILLARS)
    assert v["verified"] >= 2
    assert any(abs(u["value"] - 999.5) < 0.01 for u in v["unverified_details"])
    assert v["verdict"] == "REVIEW"


def test_verification_passes_grounded_text():
    text = "Revenue growth of 18.0% with a 22.0% net margin and 11.0x EV/EBITDA."
    v = verify.verify_claims(text, PILLARS)
    assert v["verdict"] == "PASS"
    assert v["grounding_rate"] == 1.0


def test_every_computed_metric_is_scored_or_explicitly_exempt():
    """Coverage guard. A metric that is computed and cited by agents but absent
    from the signal table is silently ignored by the score: operating_margin_pct
    was in exactly that state, so a 0.34%-vs-11.48% error changed the narrative
    and left the verdict untouched.

    Signal names are shorter than metric names, so the mapping is explicit here.
    Anything new must be mapped to a signal or added to the exempt set with a
    reason, which makes "we forgot to score this" impossible to do by accident.
    """
    import metrics

    metric_to_signal = {
        # fundamentals
        "revenue_growth_yoy_pct": "revenue_growth",
        "operating_margin_pct": "operating_margin",
        "net_margin_pct": "net_margin",
        "fcf_margin_pct": "fcf_margin",
        "cash_conversion_ocf_over_ni": "cash_conversion",
        "debt_to_equity": "leverage",
        "roe_pct": "roe",
        # valuation
        "pe_ttm": "pe_ttm", "pe_forward": "pe_forward", "pb": "pb",
        "ps_ttm": "ps_ttm", "ev_to_ebitda": "ev_to_ebitda",
        # technicals
        "trend_vs_sma200": "trend_sma200", "trend_vs_sma50": "trend_sma50",
        "rsi14": "rsi14", "pct_from_52w_high": "from_52w_high",
        # risk
        "annualised_vol_pct": "volatility", "max_drawdown_pct": "max_drawdown",
        "beta": "beta",
        # governance
        "held_by_institutions_pct": "institutional_holding",
        "held_by_insiders_pct": "insider_holding",
        "share_dilution_pct": "dilution",
    }
    exempt = {
        "fundamentals.revenue": "input to margins",
        "fundamentals.net_income": "input to margins and FCF",
        "fundamentals.fcf": "input to fcf_margin",
        "fundamentals.gross_margin_pct": "no signal yet; add one before trusting it",
        "fundamentals.roa_pct": "redundant with ROE and leverage",
        "fundamentals.cash_to_assets_pct": "redundant with leverage",
        "fundamentals.net_debt_to_ebitda": "not populated yet",
        "valuation.market_cap": "context only",
        "valuation.ev_to_revenue": "redundant with EV/EBITDA",
        "valuation.peg": "input to valuation commentary",
        "valuation.dividend_yield_pct": "not scored, income not in the rubric",
        "valuation.fifty_day_avg": "context only",
        "valuation.two_hundred_day_avg": "context only",
        "technicals.last_close": "input to trend signals",
        "technicals.sma50": "input to trend signals",
        "technicals.sma200": "input to trend signals",
        "technicals.high_52w": "input to from_52w_high",
        "technicals.low_52w": "input to from_52w_high",
        "technicals.atr14_pct": "not scored yet",
        "technicals.vol_ann_pct": "scored under risk.volatility",
        "technicals.max_drawdown_pct": "scored under risk.max_drawdown",
        "technicals.volume_vs_20d_avg": "not scored yet",
        "risk.downside_deviation_pct": "redundant with volatility",
        "risk.net_debt_to_equity": "not populated yet",
        "risk.share_dilution_pct": "scored under governance.dilution",
        "risk.held_by_institutions_pct": "scored under governance.institutional_holding",
        "risk.short_ratio": "not scored yet",
        "governance.audit_risk": "not scored yet",
        "governance.overall_risk": "not scored yet",
        "governance.governance_risk": "not populated, schema drift",
        "governance.compensation_risk": "not scored yet",
        "governance.shares_outstanding": "input to dilution",
    }

    sig = rubric.build_signals({})
    # A signal may live in a different pillar than its source metric
    # (technicals.max_drawdown_pct is scored as risk.max_drawdown), so search all.
    all_signals = {s for group in sig.values() for s in group}
    unmapped = []
    for pillar, keys in metrics.KEYS_BY_PILLAR.items():
        for k in keys:
            path = f"{pillar}.{k}"
            target = metric_to_signal.get(k)
            if target and target in all_signals:
                continue
            if path in exempt:
                continue
            unmapped.append(path)
    assert not unmapped, f"computed but neither scored nor exempt: {unmapped}"

    # and the mapping must not point at signals that do not exist
    for pillar, keys in metrics.KEYS_BY_PILLAR.items():
        for k in keys:
            t = metric_to_signal.get(k)
            if t:
                assert t in all_signals, f"{pillar}.{k} -> missing signal {t}"


def test_citation_checker_accepts_provenance_keys_and_rejects_invented_ones():
    """The bundle exposes top-level `pillar_coverage`, so citing it is legitimate.
    It was previously absent from the allowlist, so agents were flagged for
    referencing real provenance. Invented keys must still be rejected."""
    bundle = {"pillars": {"governance": {"beta": 1.0}}, "context": {}, "extras": {}}

    # legitimate provenance reference
    assert not agents._check_citations(
        {"cited_metrics": ["pillar_coverage.governance"]}, bundle
    )
    # legitimate bare signal name
    assert not agents._check_citations({"cited_metrics": ["beta"]}, bundle)
    # full correct path
    assert not agents._check_citations({"cited_metrics": ["governance.beta"]}, bundle)
    # stripped path, also fine
    assert not agents._check_citations({"cited_metrics": ["pillars.governance.beta"]}, bundle)

    # invented keys are still caught
    assert agents._check_citations({"cited_metrics": ["governance.promoter_pledge_pct"]}, bundle)
    assert agents._check_citations({"cited_metrics": ["pe_ratio_5y_median"]}, bundle)


def test_verification_ignores_unit_letter_inside_a_word():
    """'Beta of 1.08 means' must not be parsed as '1.08 million'."""
    v = verify.verify_claims("Beta of 1.08 means market-level exposure.", PILLARS)
    assert v["claims_found"] == 0 or not any(
        abs(u["value"] - 1_080_000) < 1 for u in v["unverified_details"]
    ), v


def test_verification_ignores_lookback_labels():
    v = verify.verify_claims(
        "Price is above the 200-day average and 4.5% under the 52-week high.", PILLARS
    )
    assert not any(abs(u["value"] - 200) < 1 for u in v["unverified_details"]), v


def test_verification_credits_sourced_extras():
    extras = {"institutional_holders": [{"holder": "Blackrock", "value": 386428980144.0}]}
    v = verify.verify_claims("Blackrock holds 386,428,980,144 in value.", PILLARS, extras=extras)
    assert v["unverified"] == 0, v


def test_b_and_t_suffixed_money_keeps_its_magnitude():
    """'$416.2B' used to parse as 416: no unit for B or T, and the lookahead then cut
    the match at the decimal point. Agents write magnitudes this way constantly."""
    assert verify.extract_numbers("Revenue of $416.2B") == [(416.2e9, "b")]
    pillars = {"fundamentals": {"revenue": 416.2e9}, "valuation": {"market_cap": 4.85e12}}
    v = verify.verify_claims("Revenue of $416.2B on a $4.85T market cap.", pillars)
    assert v["claims_found"] == 2 and v["unverified"] == 0, v


def test_iso_date_digits_are_not_claims():
    """A date is not evidence. Its month and day used to be counted, and a fragment
    like 10 then collided with an unrelated metric value."""
    v = verify.verify_claims(
        "Next earnings on 2026-10-29, seven weeks after 2026-09-13.", PILLARS
    )
    assert v["claims_found"] == 0, v


def test_window_digits_are_not_claims():
    """'the 50- and 200-day averages' names two windows. Only the second used to be
    masked, so the 50 became a claim."""
    v = verify.verify_claims("Price is below the 50- and 200-day averages.", PILLARS)
    assert v["claims_found"] == 0, v


def test_a_signed_metric_accepts_the_magnitude_an_agent_states():
    """'Shares fell 1.5%' describes a metric recorded as -1.5. Sign is not the claim."""
    v = verify.verify_claims("Shares fell 1.5% and the drawdown reached 22.0%.", PILLARS)
    assert v["unverified"] == 0, v


def test_unrelated_metrics_do_not_derive_a_value():
    """pe_ttm / revenue_growth is cross-pillar with no price or level metric in the
    pair, so 5.0 is not in the acceptable set. The same pair inside one pillar is,
    which is the difference between arithmetic and coincidence."""
    cross = {"valuation": {"pe_ttm": 40.0}, "fundamentals": {"revenue_growth_yoy_pct": 8.0}}
    v = verify.verify_claims("The multiple sits at 5.0 on revenue growth.", cross)
    assert v["unverified"] == 1, v
    same = {"valuation": {"pe_ttm": 40.0, "ev_to_ebitda": 8.0}}
    v2 = verify.verify_claims("The multiple sits at 5.0 on EBITDA.", same)
    assert v2["unverified"] == 0, v2


def test_a_ratio_is_read_as_a_share_or_a_percentage():
    """'85.5% of net income' is the ratio 0.855 written the other way round, and the
    pair can be stored in either order."""
    pillars = {"fundamentals": {"net_income": 80.0, "fcf": 68.4}}
    v = verify.verify_claims("Free cash flow is 85.5% of net income.", pillars)
    assert v["unverified"] == 0, v
    v2 = verify.verify_claims("Free cash flow is 0.86 of net income.", pillars)
    assert v2["unverified"] == 0, v2


def test_coverage_and_context_are_evidence_when_passed():
    """The citation check allows pillar_coverage and context keys, so the verifier has
    to accept them once analyze.py passes them through."""
    extras = {"pillar_coverage": {"governance": 0.375}, "news_count": 10}
    v = verify.verify_claims(
        "Governance coverage is 37.5% and 10 news items returned no headline.",
        PILLARS, extras=extras,
    )
    assert v["unverified"] == 0, v


def test_short_text_needs_the_absolute_floor():
    """17% unverified passes the ratio and still hides three invented numbers in an
    18-number paragraph, so a short text fails on the count as well."""
    pillars = {"fundamentals": {"net_margin_pct": 10.0}}
    text = ("Net margin is 10.0%. " * 15) + "Gross margin was 71.2%, ROE 88.4% and PEG 44.7%."
    v = verify.verify_claims(text, pillars)
    assert (v["claims_found"], v["unverified"]) == (18, 3), v
    assert v["verdict"] == "REVIEW", v


def test_a_long_text_is_judged_by_the_ratio_only():
    """The floor is for short texts. A 33-number run with three ungrounded numbers
    keeps the ratio rule, or every long honest run would fail on the forecasts it is
    allowed to make."""
    pillars = {"fundamentals": {"net_margin_pct": 10.0}}
    text = ("Net margin is 10.0%. " * 30) + "Gross margin was 71.2%, ROE 88.4% and PEG 44.7%."
    v = verify.verify_claims(text, pillars)
    assert (v["claims_found"], v["unverified"]) == (33, 3), v
    assert v["verdict"] == "PASS", v


def test_unverified_numbers_report_their_offsets():
    """analyze.py blames an agent by offset, so the offsets must address the text it
    was given."""
    text = "Beta is 9.9 and 7.7."
    v = verify.verify_claims(text, PILLARS)
    assert v["unverified"] == 2, v
    assert len(v["unverified_at"]) == 2, v
    for at in v["unverified_at"]:
        assert text[at].isdigit(), at


def test_a_fully_invented_paragraph_is_reviewed():
    """The headline failure, guarded. Every number here is invented and none is
    evidence or honest arithmetic on evidence, so the run must not come out clean."""
    text = (
        "Revenue grew 14.7% while gross margin held at 42.4%, leaving free cash flow of "
        "185.0bn against a market cap of 2.44trn. Shares trade at 30.86x book and 28.5x "
        "forward earnings, with a dividend yield of 4.1%. Institutions hold 71.4% and "
        "insiders 12.7%. The share count fell 7.3% over the year, beta reads 2.31, and the "
        "deepest drawdown was -48.2%."
    )
    v = verify.verify_claims(text, PILLARS)
    assert v["verdict"] == "REVIEW", v
    assert v["grounding_rate"] < 0.8, v


def test_citation_checker_rejects_invented_keys_under_allowed_prefixes():
    """Regression guard for the prefix bypass. Any key starting with context./extras./
    data_quality./pillar_coverage. used to be accepted, including the peer and consensus
    data the bundle documents as absent, so an agent could cite fabricated provenance.
    A citation is now valid only if the path resolves to something the bundle contains."""
    bundle = {
        "pillars": {"governance": {"beta": 1.0}},
        "context": {"market_cap": 1.0},
        "extras": {"news_count": 3,
                   "institutional_holders": [{"holder": "Blackrock", "value": 5.0}]},
        "pillar_coverage": {"governance": 0.5},
        "data_quality": {"price_data": "yfinance daily OHLCV"},
    }
    for invented in ("context.peer_median_pe", "extras.analyst_consensus_eps",
                     "context.industry_average_margin", "data_quality.five_year_median",
                     "pillar_coverage.peer_group", "extras.analyst_consensus"):
        assert agents._check_citations({"cited_metrics": [invented]}, bundle), invented
    for real in ("context.market_cap", "extras.news_count", "pillar_coverage.governance",
                 "data_quality.price_data", "extras.institutional_holders[0].value",
                 "governance.beta", "beta"):
        assert not agents._check_citations({"cited_metrics": [real]}, bundle), real


def test_citation_checker_accepts_a_single_string_key_not_its_characters():
    """A string-valued cited_metrics used to be iterated per character, which flagged
    every reference and silently validated none of them."""
    bundle = {"pillars": {"governance": {"beta": 1.0}}, "context": {}, "extras": {}}
    assert not agents._check_citations({"cited_metrics": "governance.beta"}, bundle)


def test_citation_checker_rejects_out_of_range_list_indices():
    """Paths are generated index-free, so an invented row index used to resolve:
    `extras.institutional_holders[99].value` passed even when the bundle held one
    holder. Agents do cite the indexed form, so the index is now validated."""
    bundle = {
        "pillars": {},
        "context": {},
        "extras": {"news_count": 1,
                   "institutional_holders": [{"holder": "Blackrock", "value": 5.0}]},
    }
    assert not agents._check_citations(
        {"cited_metrics": ["extras.institutional_holders[0].value"]}, bundle
    )
    for bad in ("extras.institutional_holders[9].value", "extras.news[0].title",
                "extras.institutional_holders[0].market_value"):
        assert agents._check_citations({"cited_metrics": [bad]}, bundle), bad


def test_citation_checker_rejects_bare_nested_leaf_names():
    """A bare name should mean a metric ("beta", "news_count"), not a leaf inside a
    list item ("value", "holder"), which would let "value" stand in for a citation."""
    bundle = {"pillars": {"governance": {"beta": 1.0}}, "context": {"market_cap": 1.0},
              "extras": {"news_count": 1,
                         "institutional_holders": [{"holder": "X", "value": 5.0}]}}
    for bad in ("value", "holder", "pct"):
        assert agents._check_citations({"cited_metrics": [bad]}, bundle), bad
    for real in ("beta", "market_cap", "news_count", "institutional_holders"):
        assert not agents._check_citations({"cited_metrics": [real]}, bundle), real


def test_agent_prose_carries_no_em_dash():
    """AGENTS.md rule 7 bans U+2014, and the pipeline rewrites it on the way in."""
    dash = "\u2014"
    nested = {
        "summary": f"Revenue grew 6.43% {dash} solid but mid-single-digit.",
        "key_points": [f"Premium valuation{dash}38.15x earnings{dash}needs proof.", "clean"],
        "score": 1.0,
    }
    out = agents.strip_em_dashes(nested)
    assert out["summary"] == "Revenue grew 6.43%, solid but mid-single-digit."
    assert out["key_points"][0] == "Premium valuation, 38.15x earnings, needs proof."
    assert out["key_points"][1] == "clean" and out["score"] == 1.0
    assert dash not in str(out)
    # The prompt asks for the same thing, because a rewrite is a fallback, not a plan.
    for prompt in (agents.AGENT_SYSTEM, agents.BULL_SYSTEM, agents.BEAR_SYSTEM, agents.JUDGE_SYSTEM):
        assert "em dash" in prompt


def test_tracked_runs_carry_no_em_dash():
    """The shipped artefacts obey rule 7 too, in every string they hold."""
    import json

    def strings(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from strings(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from strings(v)

    runs = Path(__file__).resolve().parent.parent / "runs"
    files = sorted(runs.glob("*.json"))
    assert files, f"no tracked artefacts under {runs}"
    for path in files:
        text = path.read_text()
        assert "\u2014" not in text, f"escaped em dash in {path.name}"
        assert not any("\u2014" in s for s in strings(json.loads(text))), path.name


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
