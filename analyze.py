#!/usr/bin/env python
"""EquityAnalyst: multi-agent adversarial equity research.

Pipeline: Python calculates -> agents interpret -> Bull argues -> Bear attacks ->
verification checks -> deterministic rubric scores. The rubric verdict is final.

Usage:
    EA_MODEL=glm-5.3:cloud python analyze.py RELIANCE.NS
    python analyze.py AAPL --no-llm          # deterministic only, no API cost
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import agents as agent_mod  # noqa: E402
import data as data_mod  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import rubric as rubric_mod  # noqa: E402
import verify as verify_mod  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"


def build_bundle(symbol: str, with_extras: bool = True) -> dict:
    b = data_mod.load(symbol)
    pillars = metrics_mod.all_pillars(b)
    coverage = metrics_mod.pillar_coverage(pillars)
    extras = data_mod.load_extras(symbol) if with_extras else {}

    context = {
        "last_price": b.evidence[0].value if b.evidence else None,
        "market_cap": b.quote.get("marketCap"),
        "sector": b.quote.get("sector"),
        "industry": b.quote.get("industry"),
        "currency": b.quote.get("currency"),
        "long_name": b.quote.get("longName"),
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "price_bars": int(len(b.price_history)),
    }

    # Explicit provenance contract. Without this the agents treat market-cap style
    # metadata as if it were a point-in-time series and write things like "market
    # cap declined 3.5% over the period", which the bundle cannot support.
    data_quality = {
        "price_data": "yfinance daily OHLCV, auto-adjusted; bar count in context.price_bars",
        "statement_data": "yfinance annual statements; column 0 is the most recent fiscal year, not a quarter",
        "quote_metadata": "yfinance snapshot fields (market_cap, held_by_institutions_pct, short_ratio, audit_risk); these are CURRENT values only, with no history",
        "no_time_series_for": ["market_cap", "multiples", "holdings", "institutional positioning changes"],
        "no_source_for": ["peer benchmarks", "industry averages", "analyst consensus", "index/rates/FX series", "earnings transcripts", "filing prose"],
        "instruction": "Treat quote_metadata as a single current observation. Do not compute a change over time from a field listed in no_time_series_for.",
    }

    return {
        "symbol": b.symbol,
        "market": b.market,
        "context": context,
        "pillars": pillars,
        "pillar_coverage": coverage,
        "data_quality": data_quality,
        "extras": {
            "news_count": len(extras.get("news", [])),
            "news_headlines": [n["title"] for n in extras.get("news", [])[:8]],
            "news_publishers": sorted({n["publisher"] for n in extras.get("news", []) if n["publisher"]})[:6],
            "next_earnings": (extras.get("calendar") or {}).get("Earnings Date"),
            "institutional_holders": (extras.get("holders") or {}).get("institutional", [])[:8],
            "data_quality": b.evidence_coverage,
        },
        "warnings": b.warnings + extras.get("warnings", []),
    }


def deterministic_report(bundle: dict) -> dict:
    signals = rubric_mod.build_signals(bundle["pillars"])
    horizons = {h: rubric_mod.score(signals, h) for h in ("SHORT_TERM", "LONG_TERM")}
    return {"signals": signals, "horizons": horizons}


def _result_from_dict(d: dict):
    """Rebuild an AgentResult from a saved run so stages can be resumed."""
    return agent_mod.AgentResult(
        name=d.get("agent", "?"),
        status=d.get("status", "unavailable"),
        data=d.get("data") or {},
        model=d.get("model", ""),
        tokens=d.get("tokens", 0),
        error=d.get("error", ""),
        violations=d.get("violations") or [],
    )


def run(symbol: str, model: str | None = None, no_llm: bool = False, quiet: bool = False,
        resume: str | None = None) -> dict:
    t0 = time.time()
    model = model or agent_mod.DEFAULT_MODEL
    bundle = build_bundle(symbol)
    det = deterministic_report(bundle)

    result = {
        "symbol": bundle["symbol"],
        "market": bundle["market"],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model if not no_llm else None,
        "context": bundle["context"],
        "warnings": bundle["warnings"],
        "pillar_coverage": bundle["pillar_coverage"],
        "deterministic": det,
        "agents": {},
        "debate": {},
        "verification": {},
        "resumed_from": resume,
    }

    if not quiet:
        print(f"[{bundle['symbol']}] price={bundle['context']['last_price']} "
              f"coverage={bundle['pillar_coverage']}")
        for h, s in det["horizons"].items():
            print(f"  {h}: {s['verdict']} score={s['overall_score']} confidence={s['confidence']}")

    if no_llm:
        result["elapsed_s"] = round(time.time() - t0, 1)
        return result

    prev = json.loads(Path(resume).read_text()) if resume and Path(resume).exists() else {}

    reused = {n: d for n, d in (prev.get("agents") or {}).items() if d.get("status") == "ok"}
    if reused:
        if not quiet:
            print(f"  reusing {len(reused)} ok agent(s) from {resume}")
        done = {n: _result_from_dict(d) for n, d in reused.items()}
        missing = {n: _result_from_dict({"agent": n, "status": "unavailable", "data": {},
                                         "error": "not in resume set"})
                   for n in agent_mod.SPECIALISTS if n not in done}
        specialists = {**missing, **done}
    else:
        if not quiet:
            print("  running 9 specialist agents in parallel...")
        specialists = agent_mod.run_specialists(bundle, model=model)
    ok = [n for n, r in specialists.items() if r.status == "ok"]
    result["agents"] = {n: r.as_dict() for n, r in specialists.items()}
    if not quiet:
        failed = [n for n in specialists if n not in ok]
        print(f"  agents ok={len(ok)}/9" + (f" failed={failed}" if failed else ""))

    prev_bull = (prev.get("debate") or {}).get("bull")
    if prev_bull and prev_bull.get("status") == "ok":
        if not quiet:
            print("  reusing bull case, running bear...")
        bull = _result_from_dict(prev_bull)
        compact = agent_mod._compact_bundle(bundle)
        args = (bull.data.get("arguments") or [])[:6]
        bull_view = {
            "role": "bull", "thesis": bull.data.get("thesis"),
            "arguments": [{"claim": a.get("claim"), "evidence": a.get("evidence")}
                          for a in args if isinstance(a, dict)],
            "what_would_break_this": (bull.data.get("what_would_break_this") or [])[:4],
        }
        bear = agent_mod.call_with_fallback(
            "bear", agent_mod.BEAR_SYSTEM, {**compact, "bull_case": bull_view},
            model=model, max_tokens=3200,
        )
        debate = {"bull": bull, "bear": bear}
    else:
        if not quiet:
            print("  bull vs bear debate...")
        debate = agent_mod.run_debate(bundle, specialists, model=model)
    result["debate"] = {k: v.as_dict() for k, v in debate.items()}

    # Verification: check every number any agent emitted against the evidence bundle.
    all_text = " ".join(
        json.dumps(r.data, default=str) for r in list(specialists.values()) + list(debate.values())
    )
    scores = [
        v for r in specialists.values()
        for v in (r.data.get("score"), r.data.get("confidence"))
        if isinstance(v, (int, float))
    ]
    # Coverage fractions and the price context are cited by agents and allowed by
    # the citation check, so the verifier has to accept them as evidence too.
    extras = dict(bundle.get("extras") or {})
    extras["pillar_coverage"] = bundle.get("pillar_coverage") or {}
    extras["context"] = bundle.get("context") or {}
    extras["news_count"] = len(extras.get("news") or [])
    v = verify_mod.verify_claims(
        all_text, bundle["pillars"], declared_scores=scores, extras=extras,
    )
    v["agent_violations"] = {
        n: r.violations for n, r in specialists.items() if r.violations
    }
    v["failed_agents"] = [n for n, r in specialists.items() if r.status != "ok"]
    result["verification"] = v

    if not quiet:
        print(f"  verification: {v['verdict']} grounding={v['grounding_rate']} "
              f"({v['verified']}/{v['claims_found']} numbers matched)")
        if v["agent_violations"]:
            print(f"  citation violations: {list(v['agent_violations'])}")
        print("  judge synthesis...")
    judge = agent_mod.run_judge(bundle, specialists, debate, v, det["horizons"], model=model)
    result["judge"] = judge.as_dict()

    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-agent adversarial equity research")
    ap.add_argument("symbol")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-llm", action="store_true", help="deterministic-only dry run")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="reuse ok stages from a prior run json")
    args = ap.parse_args()

    res = run(args.symbol, model=args.model, no_llm=args.no_llm, quiet=args.json, resume=args.resume)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"wrote {args.out}")

    if args.json:
        print(json.dumps(res, indent=2, default=str))
    elif not args.out:
        RUNS.mkdir(exist_ok=True)
        p = RUNS / f"{res['symbol'].replace('.', '_')}_{res['generated_at'].replace(':', '')}.json"
        p.write_text(json.dumps(res, indent=2, default=str))
        print(f"\nwrote {p}")
        print(f"elapsed {res['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
