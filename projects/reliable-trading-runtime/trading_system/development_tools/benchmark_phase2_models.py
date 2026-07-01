from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_sharp import run_candidate  # type: ignore
from trading_system.runtime_engine.modeling.prop_scoring import PropScoringConfig, evaluate_prop_candidate


REQUIRED_THRESHOLDS = ("p_setup", "p_long", "p_short")
SLIPPAGE_LEVELS = (0.0, 1.0, 2.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark and rank Phase-2 model candidates.")
    p.add_argument("--csv", default=str(REPO_ROOT / "data" / "intraday" / "es" / "ES6.csv"))
    p.add_argument("--artifact-root", default=str(ROOT / "artifacts" / "phase2" / "candidates"))
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "phase2_benchmark"))
    p.add_argument("--baseline-tag", default="retrain_v6_pass2_grid_02")
    p.add_argument("--legacy-baseline-tag", default="retrain_v4")
    p.add_argument("--tags", nargs="*", default=None, help="Explicit candidate tags to benchmark.")
    p.add_argument("--tag-prefix", action="append", default=[], help="Candidate directory prefix to include. Repeatable.")
    p.add_argument("--scan-all", action="store_true", help="Include every manifest under --artifact-root.")
    p.add_argument("--test-start", default="2025-11-01")
    p.add_argument("--test-end", default="2026-01-16")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--trade-window-start", default="00:00")
    p.add_argument("--trade-window-end", default="23:59")
    p.add_argument("--max-hold-bars", type=int, default=24)
    p.add_argument("--commission-per-contract", type=float, default=2.0)
    p.add_argument("--min-trades-floor", type=int, default=20)
    p.add_argument("--score-mode", choices=["risk_adjusted", "prop_aggressive"], default="risk_adjusted")
    return p.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _manifest_path(artifact_root: Path, tag: str) -> Path:
    return artifact_root / tag / "manifest.json"


def _load_manifest(path: Path, tag: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "manifest_missing"
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc}"
    if not isinstance(payload, dict):
        return None, "manifest_not_object"
    payload.setdefault("tag", tag or path.parent.name)
    payload["_manifest_path"] = str(path.resolve())
    payload["_manifest_dir"] = str(path.parent.resolve())
    return payload, None


def _resolve_model_path(manifest: Mapping[str, Any], key: str) -> Optional[Path]:
    raw = manifest.get(key)
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    manifest_dir = Path(str(manifest.get("_manifest_dir") or "."))
    return (manifest_dir / path).resolve()


def _thresholds(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    thresholds = manifest.get("thresholds") or {}
    return dict(thresholds) if isinstance(thresholds, Mapping) else {}


def _deployability_errors(manifest: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    thresholds = _thresholds(manifest)
    missing = [key for key in REQUIRED_THRESHOLDS if thresholds.get(key) is None]
    if bool(manifest.get("rejected")):
        errors.append(f"rejected={manifest.get('rejected_reason') or 'true'}")
    if missing:
        errors.append("missing_thresholds=" + ",".join(missing))
    for key in ("setup_model_path", "dir_model_path"):
        path = _resolve_model_path(manifest, key)
        if path is None:
            errors.append(f"missing_{key}")
        elif not path.exists():
            errors.append(f"missing_artifact_{key}={path}")
    return errors


def _discover_tags(args: argparse.Namespace, artifact_root: Path) -> List[str]:
    tags: List[str] = []
    for tag in (args.baseline_tag, args.legacy_baseline_tag):
        if tag and tag not in tags:
            tags.append(str(tag))
    for tag in args.tags or []:
        if tag and tag not in tags:
            tags.append(str(tag))
    if args.scan_all or args.tag_prefix:
        for path in sorted(artifact_root.iterdir() if artifact_root.exists() else []):
            if not path.is_dir():
                continue
            name = path.name
            if args.scan_all or any(name.startswith(prefix) for prefix in args.tag_prefix):
                if name not in tags:
                    tags.append(name)
    return tags


def _sharp_args(args: argparse.Namespace, manifest_path: Path, slippage: float) -> SimpleNamespace:
    return SimpleNamespace(
        tag=None,
        manifest=str(manifest_path),
        csv=args.csv,
        instrument=args.instrument,
        contracts=args.contracts,
        trade_window_start=args.trade_window_start,
        trade_window_end=args.trade_window_end,
        max_hold_bars=args.max_hold_bars,
        out_dir=str(Path(args.out_dir).expanduser().resolve() / "_sharp"),
        p_setup=None,
        p_long=None,
        p_short=None,
        start_at=args.test_start,
        end_at=args.test_end,
        commission_per_contract=args.commission_per_contract,
        slippage_ticks=slippage,
        skip_store=True,
    )


def _max_consecutive_losses(trades: Iterable[Mapping[str, Any]]) -> int:
    streak = 0
    max_streak = 0
    for trade in trades:
        pnl = float(trade.get("pnl_usd") or 0.0)
        if pnl < 0.0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _slippage_summary(sim: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "total_pnl_usd": float(sim.get("total_pnl_usd") or 0.0),
        "profit_factor": float(sim.get("profit_factor") or 0.0),
        "max_drawdown": float(sim.get("max_drawdown") or 0.0),
        "sharpe": float(sim.get("sharpe") or 0.0),
        "trade_count": int(sim.get("trade_count") or len(sim.get("trades") or [])),
        "win_rate": float(sim.get("win_rate") or 0.0),
        "flip_rate_per_day": float(sim.get("flip_rate_per_day") or 0.0),
        "trades_per_day": float(sim.get("trades_per_day") or 0.0),
        "avg_trade_usd": float(sim.get("avg_trade_usd") or 0.0),
    }


def _failure_family(manifest: Optional[Mapping[str, Any]], deployability_errors: List[str], eval_error: Optional[str]) -> str:
    if eval_error:
        return "eval_error"
    if manifest is None:
        return "incomplete"
    reason = str(manifest.get("rejected_reason") or "").lower()
    diagnostics = manifest.get("threshold_diagnostics") or {}
    if any("missing_artifact" in item or item.startswith("missing_setup_model_path") or item.startswith("missing_dir_model_path") for item in deployability_errors):
        return "incomplete"
    if ("low trade" in reason or "min_trades" in reason) and "flip" not in reason:
        return "low_trades"
    if "flip" in reason and "min_trades" not in reason and "low trade" not in reason:
        return "high_flip"
    if "bad fade" in reason or "bad_fade" in reason:
        return "bad_fade"
    if isinstance(diagnostics, Mapping):
        counts = {
            "low_trades": int(diagnostics.get("rejected_low_trades") or 0),
            "high_flip": int(diagnostics.get("rejected_high_flip") or 0),
            "bad_fade": int(diagnostics.get("rejected_bad_fade") or 0),
        }
        winner = max(counts, key=counts.get)
        if counts[winner] > 0:
            return winner
    if bool(manifest.get("rejected")):
        return "threshold_rejected"
    if deployability_errors:
        return "not_deployable"
    return "none"


def _trainer_hint(failure_family: str, manifest: Optional[Mapping[str, Any]], row: Optional[Mapping[str, Any]] = None) -> str:
    if failure_family == "low_trades":
        return "Lower setup multiplier or label threshold; keep p_setup search wide/lower."
    if failure_family == "high_flip":
        return "Raise direction thresholds, add churn penalty, or try a longer horizon."
    if failure_family == "bad_fade":
        return "Tighten fade/countertrend filters before widening entries."
    if failure_family == "eval_error":
        return "Fix benchmark/runtime compatibility before using this tag."
    if failure_family == "incomplete":
        return "Finish training artifacts and manifest before replay/live use."
    if failure_family == "threshold_rejected":
        metrics = ((manifest or {}).get("metrics") or {}) if isinstance((manifest or {}).get("metrics"), Mapping) else {}
        setup = metrics.get("setup") or {}
        auc = float(setup.get("roc_auc_test") or 0.0) if isinstance(setup, Mapping) else 0.0
        if auc >= 0.75:
            return "Setup model separates well; relax threshold search/gates before changing features."
        return "Try broader labels or less selective setup configs."
    if row:
        slip1 = (row.get("slippage") or {}).get("slip_1") or {}
        if row.get("deployable") and not row.get("slippage_pass"):
            return "Reduce low-edge churn and require stronger EV per trade."
        if float(slip1.get("total_pnl_usd") or 0.0) <= 0.0:
            return "Keep deployability, but retune thresholds/close overlay for positive OOS PnL."
    return "Keep as reference; no immediate trainer bias from this row."


def _risk_score(row: Mapping[str, Any], min_trades_floor: int) -> float:
    slip1 = (row.get("slippage") or {}).get("slip_1") or {}
    bad_fade = float(row.get("bad_fade_rate") or 0.0)
    max_losses = int(row.get("max_consecutive_losses") or 0)
    trade_count = int(slip1.get("trade_count") or 0)
    slippage_penalty = 0.0 if row.get("slippage_pass") else 250.0
    low_trade_penalty = max(0, min_trades_floor - trade_count) * 8.0
    return (
        float(slip1.get("sharpe") or 0.0) * 100.0
        + float(slip1.get("profit_factor") or 0.0) * 50.0
        + float(slip1.get("total_pnl_usd") or 0.0) / 100.0
        + min(trade_count, 200) * 0.5
        - abs(float(slip1.get("max_drawdown") or 0.0)) / 100.0
        - max_losses * 10.0
        - float(slip1.get("flip_rate_per_day") or 0.0) * 25.0
        - bad_fade * 100.0
        - slippage_penalty
        - low_trade_penalty
    )


def _evaluate_deployable(args: argparse.Namespace, manifest: Mapping[str, Any], manifest_path: Path) -> Dict[str, Any]:
    slippage: Dict[str, Any] = {}
    trades_for_loss: List[Mapping[str, Any]] = []
    for level in SLIPPAGE_LEVELS:
        result = run_candidate(_sharp_args(args, manifest_path, level))
        sim = result.get("sim") or {}
        slippage[f"slip_{level:g}"] = _slippage_summary(sim)
        if abs(level - 1.0) < 1e-9:
            trades_for_loss = list(sim.get("trades") or [])

    slippage_pass = all(
        int(payload.get("trade_count") or 0) >= max(1, int(args.min_trades_floor) // 2)
        and float(payload.get("profit_factor") or 0.0) >= 1.0
        and float(payload.get("total_pnl_usd") or 0.0) > 0.0
        for payload in slippage.values()
    )
    behavior = manifest.get("behavior_audit_test") or {}
    bad_fade = float(behavior.get("countertrend_rate") or 0.0) if isinstance(behavior, Mapping) else 0.0
    result = {
        "slippage": slippage,
        "slippage_pass": slippage_pass,
        "max_consecutive_losses": _max_consecutive_losses(trades_for_loss),
        "bad_fade_rate": bad_fade,
    }
    if getattr(args, "score_mode", "risk_adjusted") == "prop_aggressive":
        result["prop_evaluation"] = evaluate_prop_candidate(
            slip1=slippage.get("slip_1") or {},
            slippage=slippage,
            trades=trades_for_loss,
            bad_fade_rate=bad_fade,
            config=PropScoringConfig(),
        )
    return result


def _base_row(tag: str, manifest_path: Path, manifest: Optional[Mapping[str, Any]], status: str) -> Dict[str, Any]:
    return {
        "tag": tag,
        "manifest_path": str(manifest_path),
        "status": status,
        "deployable": False,
        "rejected": bool((manifest or {}).get("rejected")) if manifest else False,
        "thresholds": _thresholds(manifest or {}),
        "threshold_diagnostics": (manifest or {}).get("threshold_diagnostics") or {},
        "rejected_reason": (manifest or {}).get("rejected_reason") if manifest else None,
        "config": (manifest or {}).get("config") or {},
        "slippage": {},
        "slippage_pass": False,
        "max_consecutive_losses": None,
        "bad_fade_rate": None,
        "prop_evaluation": None,
        "score": float("-inf"),
        "tier": "D",
        "rank": None,
        "failure_family": "none",
        "trainer_hint": "",
        "eval_error": None,
        "deployability_errors": [],
    }


def _evaluate_tag(args: argparse.Namespace, artifact_root: Path, tag: str) -> Dict[str, Any]:
    path = _manifest_path(artifact_root, tag)
    manifest, load_error = _load_manifest(path, tag)
    row = _base_row(tag, path, manifest, "loaded" if manifest else "invalid")
    if load_error:
        row["status"] = "invalid" if path.exists() else "incomplete"
        row["eval_error"] = load_error
        row["failure_family"] = _failure_family(None, [], load_error)
        row["trainer_hint"] = _trainer_hint(row["failure_family"], None)
        return row

    errors = _deployability_errors(manifest or {})
    row["deployability_errors"] = errors
    if errors:
        row["tier"] = "C" if bool((manifest or {}).get("rejected")) else "D"
        row["failure_family"] = _failure_family(manifest, errors, None)
        row["trainer_hint"] = _trainer_hint(row["failure_family"], manifest, row)
        return row

    row["deployable"] = True
    try:
        row.update(_evaluate_deployable(args, manifest or {}, path))
    except Exception as exc:
        row["deployable"] = False
        row["tier"] = "D"
        row["status"] = "eval_failed"
        row["eval_error"] = str(exc)
        row["failure_family"] = _failure_family(manifest, errors, str(exc))
        row["trainer_hint"] = _trainer_hint(row["failure_family"], manifest, row)
        return row

    slip1 = (row.get("slippage") or {}).get("slip_1") or {}
    strong = (
        bool(row.get("slippage_pass"))
        and float(slip1.get("total_pnl_usd") or 0.0) > 0.0
        and float(slip1.get("profit_factor") or 0.0) >= 1.0
    )
    row["tier"] = "A" if strong else "B"
    row["status"] = "benchmarked"
    prop_eval = row.get("prop_evaluation") or {}
    if getattr(args, "score_mode", "risk_adjusted") == "prop_aggressive" and isinstance(prop_eval, Mapping):
        row["score"] = float(prop_eval.get("score") or float("-inf"))
        if not bool(prop_eval.get("deployable")):
            row["tier"] = "B" if row["tier"] == "A" else row["tier"]
    else:
        row["score"] = _risk_score(row, int(args.min_trades_floor))
    row["failure_family"] = _failure_family(manifest, [], None)
    row["trainer_hint"] = _trainer_hint(row["failure_family"], manifest, row)
    return row


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    rows.sort(key=lambda row: (tier_order.get(str(row.get("tier")), 9), -float(row.get("score") or float("-inf")), str(row.get("tag"))))
    rank = 1
    for row in rows:
        if row.get("tier") in ("A", "B"):
            row["rank"] = rank
            rank += 1
    return rows


def _compact_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    slip1 = (row.get("slippage") or {}).get("slip_1") or {}
    return {
        "tag": row.get("tag"),
        "tier": row.get("tier"),
        "deployable": row.get("deployable"),
        "rejected": row.get("rejected"),
        "rank": row.get("rank"),
        "score": row.get("score"),
        "pnl_slip1": slip1.get("total_pnl_usd"),
        "profit_factor_slip1": slip1.get("profit_factor"),
        "sharpe_slip1": slip1.get("sharpe"),
        "max_dd_slip1": slip1.get("max_drawdown"),
        "trade_count_slip1": slip1.get("trade_count"),
        "win_rate_slip1": slip1.get("win_rate"),
        "flip_rate_slip1": slip1.get("flip_rate_per_day"),
        "trades_per_day_slip1": slip1.get("trades_per_day"),
        "slippage_pass": row.get("slippage_pass"),
        "max_consecutive_losses": row.get("max_consecutive_losses"),
        "prop_deployable": ((row.get("prop_evaluation") or {}).get("deployable") if isinstance(row.get("prop_evaluation"), Mapping) else None),
        "prop_errors": ";".join((row.get("prop_evaluation") or {}).get("errors") or []) if isinstance(row.get("prop_evaluation"), Mapping) else None,
        "rejected_reason": row.get("rejected_reason"),
        "failure_family": row.get("failure_family"),
        "trainer_hint": row.get("trainer_hint"),
    }


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = [_compact_row(row) for row in rows]
    if not compact:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(compact[0].keys()))
        writer.writeheader()
        writer.writerows(compact)


def _format_num(value: Any, digits: int = 2) -> str:
    try:
        num = float(value)
    except Exception:
        return ""
    if not math.isfinite(num):
        return ""
    return f"{num:.{digits}f}"


def _write_markdown(path: Path, rows: List[Mapping[str, Any]], guidance: Mapping[str, Any], args: argparse.Namespace) -> None:
    lines: List[str] = []
    lines.append("# Phase2 Benchmark")
    lines.append("")
    lines.append(f"- Test window: `{args.test_start}` to `{args.test_end}`")
    lines.append(f"- Deployment baseline: `{args.baseline_tag}`")
    lines.append(f"- Legacy baseline: `{args.legacy_baseline_tag}`")
    lines.append(f"- Score mode: `{args.score_mode}`")
    lines.append(f"- Best deployable: `{guidance.get('best_deployable_tag') or 'none'}`")
    lines.append(f"- Best rejected learning tag: `{guidance.get('best_rejected_learning_tag') or 'none'}`")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| tier | rank | tag | score | pnl@1 | pf@1 | sharpe@1 | dd@1 | trades@1 | t/day@1 | prop | slippage | failure |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for row in rows:
        slip1 = (row.get("slippage") or {}).get("slip_1") or {}
        prop_eval = row.get("prop_evaluation") or {}
        lines.append(
            "| {tier} | {rank} | `{tag}` | {score} | {pnl} | {pf} | {sharpe} | {dd} | {trades} | {tpd} | {prop} | {slip} | {failure} |".format(
                tier=row.get("tier"),
                rank=row.get("rank") or "",
                tag=row.get("tag"),
                score=_format_num(row.get("score")),
                pnl=_format_num(slip1.get("total_pnl_usd")),
                pf=_format_num(slip1.get("profit_factor")),
                sharpe=_format_num(slip1.get("sharpe")),
                dd=_format_num(slip1.get("max_drawdown")),
                trades=slip1.get("trade_count") if slip1 else "",
                tpd=_format_num(slip1.get("trades_per_day")),
                prop="pass" if isinstance(prop_eval, Mapping) and prop_eval.get("deployable") else ("fail" if prop_eval else ""),
                slip="pass" if row.get("slippage_pass") else "fail",
                failure=row.get("failure_family") or "",
            )
        )
    lines.append("")
    lines.append("## Trainer Guidance")
    lines.append("")
    for item in guidance.get("recommended_next_biases") or []:
        lines.append(f"- {item}")
    avoid = guidance.get("avoid_next_biases") or []
    if avoid:
        lines.append("")
        lines.append("## Avoid Next")
        lines.append("")
        for item in avoid:
            lines.append(f"- {item}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _guidance(rows: List[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    tier_ab = [row for row in rows if row.get("tier") in ("A", "B")]
    tier_a = [row for row in rows if row.get("tier") == "A"]
    rejected = [row for row in rows if row.get("tier") == "C"]
    failures: Dict[str, int] = {}
    for row in rows:
        family = str(row.get("failure_family") or "none")
        failures[family] = failures.get(family, 0) + 1

    recommendations: List[str] = []
    avoid: List[str] = []
    if failures.get("low_trades", 0) > 0:
        recommendations.append("Bias next generation toward lower setup multipliers, lower label thresholds, and wider/lower p_setup search.")
    if failures.get("high_flip", 0) > 0:
        recommendations.append("Bias next generation toward higher direction thresholds, longer horizons, or explicit churn penalties.")
    if failures.get("bad_fade", 0) > 0:
        recommendations.append("Bias next generation toward stricter fade/countertrend filtering before adding more entries.")
    if any(row.get("deployable") and not row.get("slippage_pass") for row in rows):
        recommendations.append("For deployable slippage failures, reduce low-edge churn and require stronger EV per trade.")
    if not tier_a and tier_ab:
        recommendations.append("Keep deployable structures, but tune thresholds/close overlay until OOS PnL and slippage pass.")
    if not tier_ab:
        recommendations.append("Do not promote new candidates; keep the deployment baseline until a candidate has thresholds and passes OOS.")
    if failures.get("threshold_rejected", 0) > failures.get("low_trades", 0) + failures.get("high_flip", 0):
        avoid.append("Avoid changing model families first; inspect threshold diagnostics and gate strictness first.")
    if failures.get("low_trades", 0) >= max(2, len(rejected) // 2):
        avoid.append("Avoid raising setup multipliers until trade count recovers.")
    if getattr(args, "score_mode", "risk_adjusted") == "prop_aggressive":
        prop_failed = [
            row for row in tier_ab
            if isinstance(row.get("prop_evaluation"), Mapping) and not row["prop_evaluation"].get("deployable")
        ]
        if prop_failed:
            recommendations.append("For prop-aggressive mode, prefer candidates with 4-10 trades/day, no daily-loss breaches, and loss streak <= 3 over raw PnL leaders.")

    best_deployable = tier_ab[0]["tag"] if tier_ab else None
    best_rejected = rejected[0]["tag"] if rejected else None
    baseline = next((row for row in rows if row.get("tag") == args.baseline_tag), None)
    legacy = next((row for row in rows if row.get("tag") == args.legacy_baseline_tag), None)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_deployable_tag": best_deployable,
        "best_rejected_learning_tag": best_rejected,
        "failure_counts": dict(sorted(failures.items())),
        "recommended_next_biases": recommendations,
        "avoid_next_biases": avoid,
        "baseline_comparison": {
            "deployment_baseline_tag": args.baseline_tag,
            "legacy_baseline_tag": args.legacy_baseline_tag,
            "deployment_baseline": _compact_row(baseline) if baseline else None,
            "legacy_baseline": _compact_row(legacy) if legacy else None,
        },
        "score_mode": getattr(args, "score_mode", "risk_adjusted"),
    }


def main() -> int:
    args = _parse_args()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tags = _discover_tags(args, artifact_root)
    rows = [_evaluate_tag(args, artifact_root, tag) for tag in tags]
    rows = _sort_rows(rows)
    guidance = _guidance(rows, args)

    payload = {
        "generated_at": guidance["generated_at"],
        "artifact_root": str(artifact_root),
        "csv": args.csv,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "baseline_tag": args.baseline_tag,
        "legacy_baseline_tag": args.legacy_baseline_tag,
        "score_mode": args.score_mode,
        "rows": rows,
        "guidance": guidance,
    }
    _write_json(out_dir / "phase2_benchmark.json", payload)
    _write_json(out_dir / "trainer_guidance.json", guidance)
    _write_csv(out_dir / "phase2_benchmark.csv", rows)
    _write_markdown(out_dir / "phase2_benchmark.md", rows, guidance, args)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "rows": len(rows),
                "best_deployable_tag": guidance.get("best_deployable_tag"),
                "best_rejected_learning_tag": guidance.get("best_rejected_learning_tag"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
