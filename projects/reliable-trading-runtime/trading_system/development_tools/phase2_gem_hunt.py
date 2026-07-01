from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE2_ROOT = ROOT / "artifacts" / "phase2"


@dataclass
class RuntimeContext:
    active_phase2_tag: Optional[str]
    champion_tag: Optional[str]
    champion_deployment_baseline_tag: Optional[str]
    champion_baseline_tag: Optional[str]
    champion_path: Optional[str]


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:  # NaN
        return None
    return out


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _load_json_any(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_active_phase2_tag(master_yaml_path: Path) -> Optional[str]:
    if not master_yaml_path.exists():
        return None
    for line in master_yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("phase2_tag:"):
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                tag = parts[1].strip().strip("'\"")
                if tag:
                    return tag
    return None


def _read_runtime_context(phase2_root: Path, master_yaml_path: Path) -> RuntimeContext:
    active_phase2_tag = _read_active_phase2_tag(master_yaml_path)
    champion_path = phase2_root / "candidates" / "current_champion.json"
    champion_payload = _load_json(champion_path) or {}
    return RuntimeContext(
        active_phase2_tag=active_phase2_tag,
        champion_tag=str(champion_payload.get("tag") or "") or None,
        champion_deployment_baseline_tag=str(champion_payload.get("deployment_baseline_tag") or "") or None,
        champion_baseline_tag=str(champion_payload.get("baseline_tag") or "") or None,
        champion_path=str(champion_path.resolve()) if champion_path.exists() else None,
    )


def _candidate_context(phase2_root: Path, candidate_tag: str) -> Dict[str, Any]:
    candidate_dir = phase2_root / "candidates" / candidate_tag
    manifest_path = candidate_dir / "manifest.json"
    benchmark_path = candidate_dir / "benchmark_results.json"
    quality_path = candidate_dir / "quality_gates.json"
    manifest = _load_json(manifest_path)
    benchmark = _load_json(benchmark_path)
    quality = _load_json_any(quality_path)
    has_benchmark = bool(benchmark and isinstance(benchmark.get("metrics"), dict))
    bench_metrics = benchmark.get("metrics", {}) if isinstance(benchmark, dict) else {}
    pnl_new = _safe_float((bench_metrics.get("pnl_slip1") or {}).get("new"))
    sharpe_new = _safe_float((bench_metrics.get("sharpe_slip1") or {}).get("new"))
    trades_new = _safe_float((bench_metrics.get("trade_count") or {}).get("new"))
    win_new = _safe_float((bench_metrics.get("win_rate") or {}).get("new"))
    pf_new = _safe_float((bench_metrics.get("profit_factor") or {}).get("new"))

    deployability_errors: List[str] = []
    is_deployable = False
    if manifest is None:
        deployability_errors.append("manifest_missing")
    else:
        if bool(manifest.get("rejected")):
            deployability_errors.append(f"manifest_rejected:{manifest.get('rejected_reason') or 'true'}")
        thresholds = manifest.get("thresholds") if isinstance(manifest.get("thresholds"), dict) else {}
        for key in ("p_setup", "p_long", "p_short"):
            if thresholds.get(key) is None:
                deployability_errors.append(f"missing_threshold:{key}")
        for model_key in ("setup_model_path", "dir_model_path"):
            rel = manifest.get(model_key)
            if not rel:
                deployability_errors.append(f"missing_{model_key}")
                continue
            abs_path = (candidate_dir / str(rel)).resolve()
            if not abs_path.exists():
                deployability_errors.append(f"missing_file:{model_key}")
        if not deployability_errors:
            is_deployable = True

    quality_pass = None
    if isinstance(quality, list):
        failures = [x for x in quality if isinstance(x, dict) and (not bool(x.get("pass")))]
        quality_pass = len(failures) == 0
    elif isinstance(quality, dict):
        quality_pass = bool(quality.get("quality_gates_passed"))

    return {
        "candidate_tag": candidate_tag,
        "candidate_dir": str(candidate_dir),
        "has_manifest": manifest is not None,
        "has_benchmark": has_benchmark,
        "has_quality_gates": quality is not None,
        "is_deployable": is_deployable,
        "deployability_errors": deployability_errors,
        "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else None,
        "benchmark_path": str(benchmark_path.resolve()) if benchmark_path.exists() else None,
        "quality_gates_path": str(quality_path.resolve()) if quality_path.exists() else None,
        "benchmark_pnl_slip1_new": pnl_new,
        "benchmark_sharpe_slip1_new": sharpe_new,
        "benchmark_trade_count_new": trades_new,
        "benchmark_win_rate_new": win_new,
        "benchmark_profit_factor_new": pf_new,
        "quality_gates_pass": quality_pass,
    }


def _model_family(rel_dir: Path) -> str:
    parts = rel_dir.parts
    if not parts:
        return "unknown"
    if parts[0] == "candidates" and len(parts) >= 2:
        return parts[1]
    return parts[0]


def _as_bool(value: Any) -> bool:
    return bool(value)


def _row_action(row: Dict[str, Any]) -> Tuple[str, str]:
    robust = _as_bool(row.get("robust_pass"))
    has_bench = _as_bool(row.get("has_benchmark"))
    deployable = _as_bool(row.get("is_deployable"))
    pnl = _safe_float(row.get("benchmark_pnl_slip1_new")) or 0.0
    sharpe = _safe_float(row.get("benchmark_sharpe_slip1_new")) or 0.0
    trades = _safe_float(row.get("benchmark_trade_count_new")) or 0.0
    ll = _safe_float(row.get("dir_log_loss_val")) or 9.99
    low_conf = _as_bool(row.get("high_metric_low_confidence"))
    if robust and has_bench:
        if pnl >= 100000.0 and sharpe >= 15.0 and trades >= 500.0:
            return "promote now", "benchmark-backed and passes practical deployment gate profile"
        if pnl > 0.0 and ll <= 0.58:
            return "package then benchmark", "already benchmarked but below hard gate targets; keep in promotion loop"
        return "archive", "benchmark evidence is weak relative to robust alternatives"
    if robust and (not deployable or not has_bench):
        return "package then benchmark", "strong classifier metrics but missing deployable benchmark proof"
    if low_conf:
        return "archive", "high headline metrics with low-confidence holdout size"
    return "archive", "fails strict robustness gate"


def _robustness(row: Dict[str, Any], min_n_val: int, min_n_test: int) -> Tuple[bool, List[str], bool]:
    reasons: List[str] = []
    n_val = _safe_int(row.get("dir_n_val")) or 0
    n_test = _safe_int(row.get("dir_n_test")) or 0
    n_train = _safe_int(row.get("dir_n_train")) or 0
    for key in ("dir_log_loss_val", "dir_log_loss_test", "dir_roc_auc_val", "dir_roc_auc_test"):
        if _safe_float(row.get(key)) is None:
            reasons.append(f"missing_metric:{key}")
    if n_val < min_n_val:
        reasons.append(f"n_val_below_min:{n_val}<{min_n_val}")
    if n_test < min_n_test:
        reasons.append(f"n_test_below_min:{n_test}<{min_n_test}")
    if n_train < max(1000, min_n_val * 3):
        reasons.append(f"n_train_low:{n_train}")
    robust_pass = len(reasons) == 0
    ll = _safe_float(row.get("dir_log_loss_val"))
    auc = _safe_float(row.get("dir_roc_auc_val"))
    high_metric_low_conf = (not robust_pass) and (
        (ll is not None and ll < 0.45) or (auc is not None and auc > 0.90)
    )
    return robust_pass, reasons, high_metric_low_conf


def _deployable_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    robust = _as_bool(row.get("robust_pass"))
    has_bench = _as_bool(row.get("has_benchmark"))
    deployable = _as_bool(row.get("is_deployable"))
    ll = _safe_float(row.get("dir_log_loss_val"))
    auc = _safe_float(row.get("dir_roc_auc_val"))
    pnl = _safe_float(row.get("benchmark_pnl_slip1_new"))
    sharpe = _safe_float(row.get("benchmark_sharpe_slip1_new"))
    trades = _safe_float(row.get("benchmark_trade_count_new"))
    if robust and has_bench:
        tier = 0
    elif robust and deployable:
        tier = 1
    elif robust:
        tier = 2
    else:
        tier = 3
    return (
        tier,
        -(pnl if pnl is not None else -1e12),
        -(sharpe if sharpe is not None else -1e12),
        -(trades if trades is not None else -1e12),
        (ll if ll is not None else 99.0),
        -(auc if auc is not None else -1.0),
    )


def _watchlist_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    ll = _safe_float(row.get("dir_log_loss_val"))
    auc = _safe_float(row.get("dir_roc_auc_val"))
    n_test = _safe_int(row.get("dir_n_test")) or 0
    is_sweep = _as_bool(row.get("is_sweep"))
    return (
        0 if is_sweep else 1,
        (ll if ll is not None else 99.0),
        -(auc if auc is not None else -1.0),
        -n_test,
    )


def _iter_model_rows(phase2_root: Path, runtime_ctx: RuntimeContext, min_n_val: int, min_n_test: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dir_metric_files = sorted(phase2_root.rglob("dir.metrics.json"))
    for metric_file in dir_metric_files:
        model_dir = metric_file.parent
        rel_dir = model_dir.relative_to(phase2_root)
        parts = rel_dir.parts
        dir_metrics = _load_json(metric_file) or {}
        setup_metrics = _load_json(model_dir / "setup.metrics.json") or {}

        candidate_tag = None
        is_sweep = False
        tag_or_trial = str(rel_dir).replace("\\", "/")
        if len(parts) >= 2 and parts[0] == "candidates":
            candidate_tag = parts[1]
            is_sweep = len(parts) >= 4 and parts[2] == "sweeps"
            tag_or_trial = candidate_tag if len(parts) == 2 else f"{candidate_tag}/{'/'.join(parts[2:])}"

        candidate_ctx = _candidate_context(phase2_root, candidate_tag) if candidate_tag else {}
        row: Dict[str, Any] = {
            "tag_or_trial": tag_or_trial,
            "source_path": str(rel_dir).replace("\\", "/"),
            "model_family": _model_family(rel_dir),
            "candidate_tag": candidate_tag,
            "is_sweep": is_sweep,
            "is_deployable": candidate_ctx.get("is_deployable", False),
            "has_benchmark": candidate_ctx.get("has_benchmark", False),
            "has_manifest": candidate_ctx.get("has_manifest", False),
            "active_phase2_tag": runtime_ctx.active_phase2_tag,
            "champion_tag": runtime_ctx.champion_tag,
            "champion_deployment_baseline_tag": runtime_ctx.champion_deployment_baseline_tag,
            "champion_baseline_tag": runtime_ctx.champion_baseline_tag,
            "champion_path": runtime_ctx.champion_path,
            "manifest_path": candidate_ctx.get("manifest_path"),
            "benchmark_path": candidate_ctx.get("benchmark_path"),
            "quality_gates_path": candidate_ctx.get("quality_gates_path"),
            "quality_gates_pass": candidate_ctx.get("quality_gates_pass"),
            "deployability_errors": ";".join(candidate_ctx.get("deployability_errors", [])),
            "dir_log_loss_val": dir_metrics.get("log_loss_val"),
            "dir_log_loss_test": dir_metrics.get("log_loss_test"),
            "dir_roc_auc_val": dir_metrics.get("roc_auc_val"),
            "dir_roc_auc_test": dir_metrics.get("roc_auc_test"),
            "dir_balanced_accuracy_val": dir_metrics.get("balanced_accuracy_val"),
            "dir_balanced_accuracy_test": dir_metrics.get("balanced_accuracy_test"),
            "dir_n_train": dir_metrics.get("n_train"),
            "dir_n_val": dir_metrics.get("n_val"),
            "dir_n_test": dir_metrics.get("n_test"),
            "dir_feature_count": dir_metrics.get("feature_count_after_filtering", dir_metrics.get("feature_count")),
            "setup_log_loss_val": setup_metrics.get("log_loss_val"),
            "setup_roc_auc_val": setup_metrics.get("roc_auc_val"),
            "benchmark_pnl_slip1_new": candidate_ctx.get("benchmark_pnl_slip1_new"),
            "benchmark_sharpe_slip1_new": candidate_ctx.get("benchmark_sharpe_slip1_new"),
            "benchmark_trade_count_new": candidate_ctx.get("benchmark_trade_count_new"),
            "benchmark_win_rate_new": candidate_ctx.get("benchmark_win_rate_new"),
            "benchmark_profit_factor_new": candidate_ctx.get("benchmark_profit_factor_new"),
        }
        robust_pass, robust_reasons, high_metric_low_conf = _robustness(row, min_n_val=min_n_val, min_n_test=min_n_test)
        row["robust_pass"] = robust_pass
        row["robustness_reasons"] = ";".join(robust_reasons)
        row["high_metric_low_confidence"] = high_metric_low_conf
        action, action_reason = _row_action(row)
        row["recommended_action"] = action
        row["action_reason"] = action_reason
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_none_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                cells.append(f"{value:.6f}" if abs(value) < 1000 else f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _report_markdown(
    runtime_ctx: RuntimeContext,
    inventory_count: int,
    deployable_gems: List[Dict[str, Any]],
    hidden_watchlist: List[Dict[str, Any]],
    checks: Dict[str, Any],
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines: List[str] = []
    lines.append("# Phase-2 Model Gem Hunt Report")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append(f"- Total models indexed (`dir.metrics.json`): **{inventory_count}**")
    lines.append(f"- Active `phase2_tag` (`trading_system/runtime_engine/runtime_config/master.yaml`): `{runtime_ctx.active_phase2_tag}`")
    lines.append(f"- Current champion tag: `{runtime_ctx.champion_tag}`")
    lines.append(f"- Champion deployment baseline tag: `{runtime_ctx.champion_deployment_baseline_tag}`")
    lines.append("")
    lines.append("## Acceptance Checks")
    lines.append("")
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Top Deployable Gems (Benchmark-backed + Robust)")
    lines.append("")
    lines.append(
        _markdown_table(
            deployable_gems[:15],
            [
                "tag_or_trial",
                "benchmark_pnl_slip1_new",
                "benchmark_sharpe_slip1_new",
                "benchmark_trade_count_new",
                "dir_log_loss_val",
                "dir_roc_auc_val",
                "recommended_action",
                "action_reason",
            ],
        )
    )
    lines.append("")
    lines.append("## Hidden Watchlist (Strong but Not Yet Deployable)")
    lines.append("")
    lines.append(
        _markdown_table(
            hidden_watchlist[:20],
            [
                "tag_or_trial",
                "is_sweep",
                "is_deployable",
                "has_benchmark",
                "dir_log_loss_val",
                "dir_roc_auc_val",
                "dir_n_val",
                "dir_n_test",
                "recommended_action",
                "action_reason",
            ],
        )
    )
    lines.append("")
    lines.append("## Promotion Summary")
    lines.append("")
    promote_now = [r for r in deployable_gems if r.get("recommended_action") == "promote now"]
    package_then_benchmark = [r for r in hidden_watchlist if r.get("recommended_action") == "package then benchmark"]
    archive = [r for r in hidden_watchlist if r.get("recommended_action") == "archive"]
    lines.append(f"- `promote now`: **{len(promote_now)}**")
    lines.append(f"- `package then benchmark`: **{len(package_then_benchmark)}**")
    lines.append(f"- `archive`: **{len(archive)}**")
    if promote_now:
        lines.append("- Promote candidates:")
        for row in promote_now[:10]:
            lines.append(f"  - `{row['tag_or_trial']}`")
    return "\n".join(lines) + "\n"


def _acceptance_checks(all_rows: List[Dict[str, Any]], phase2_root: Path, ranked_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    dir_metrics_total = len(list(phase2_root.rglob("dir.metrics.json")))
    inventory_total = len(all_rows)
    unique_sources = len({str(r.get("source_path")) for r in all_rows})
    duplicates = inventory_total - unique_sources

    robust_bench_indices = [
        idx for idx, row in enumerate(ranked_rows) if bool(row.get("robust_pass")) and bool(row.get("has_benchmark"))
    ]
    robust_nonbench_indices = [
        idx for idx, row in enumerate(ranked_rows) if bool(row.get("robust_pass")) and (not bool(row.get("has_benchmark")))
    ]
    benchmark_outrank_check = True
    if robust_bench_indices and robust_nonbench_indices:
        benchmark_outrank_check = min(robust_nonbench_indices) >= min(robust_bench_indices)

    low_conf_spikes = [
        row
        for row in ranked_rows[:20]
        if bool(row.get("high_metric_low_confidence")) and (_safe_float(row.get("dir_log_loss_val")) or 9.99) < 0.45
    ]
    low_conf_demoted_check = len(low_conf_spikes) == 0

    return {
        "completeness_dir_metrics_vs_inventory": inventory_total == dir_metrics_total,
        "completeness_counts": {"inventory_total": inventory_total, "dir_metrics_total": dir_metrics_total},
        "no_duplicate_source_paths": duplicates == 0,
        "duplicate_source_path_count": duplicates,
        "scoring_benchmark_backed_outrank_nonbenchmark": benchmark_outrank_check,
        "scoring_low_confidence_spikes_not_in_top20": low_conf_demoted_check,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-2 model gem hunt audit.")
    parser.add_argument("--phase2-root", default=str(DEFAULT_PHASE2_ROOT))
    parser.add_argument("--master-yaml", default=str(ROOT / "runtime_engine" / "runtime_config" / "master.yaml"))
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: artifacts/phase2/reviews/<timestamp>")
    parser.add_argument("--min-n-val", type=int, default=1000)
    parser.add_argument("--min-n-test", type=int, default=1000)
    parser.add_argument("--top-deployable", type=int, default=25)
    parser.add_argument("--top-watchlist", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    phase2_root = Path(args.phase2_root).expanduser().resolve()
    master_yaml = Path(args.master_yaml).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (phase2_root / "reviews" / f"gem_hunt_{timestamp}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_ctx = _read_runtime_context(phase2_root, master_yaml)
    rows = _iter_model_rows(
        phase2_root,
        runtime_ctx=runtime_ctx,
        min_n_val=int(args.min_n_val),
        min_n_test=int(args.min_n_test),
    )

    ranked = sorted(rows, key=_deployable_sort_key)
    deployable_gems = [r for r in ranked if bool(r.get("robust_pass")) and bool(r.get("has_benchmark"))][: int(args.top_deployable)]
    hidden_watchlist_candidates = [
        r
        for r in ranked
        if bool(r.get("robust_pass"))
        and (not bool(r.get("has_benchmark")) or not bool(r.get("is_deployable")))
        and (not bool(r.get("high_metric_low_confidence")))
    ]
    hidden_watchlist = sorted(hidden_watchlist_candidates, key=_watchlist_sort_key)[: int(args.top_watchlist)]

    checks = _acceptance_checks(rows, phase2_root=phase2_root, ranked_rows=ranked)

    _json_dump(output_dir / "phase2_model_inventory.json", rows)
    _write_csv(output_dir / "phase2_model_inventory.csv", rows)
    _json_dump(output_dir / "deployable_gems.json", deployable_gems)
    _write_csv(output_dir / "deployable_gems.csv", deployable_gems)
    _json_dump(output_dir / "hidden_watchlist.json", hidden_watchlist)
    _write_csv(output_dir / "hidden_watchlist.csv", hidden_watchlist)
    _json_dump(
        output_dir / "decision_summary.json",
        {
            "runtime_context": runtime_ctx.__dict__,
            "acceptance_checks": checks,
            "counts": {
                "inventory": len(rows),
                "deployable_gems": len(deployable_gems),
                "hidden_watchlist": len(hidden_watchlist),
            },
        },
    )
    report = _report_markdown(
        runtime_ctx=runtime_ctx,
        inventory_count=len(rows),
        deployable_gems=deployable_gems,
        hidden_watchlist=hidden_watchlist,
        checks=checks,
    )
    (output_dir / "phase2_gem_hunt_report.md").write_text(report, encoding="utf-8")

    print(f"OK phase2_gem_hunt complete")
    print(f"phase2_root={phase2_root}")
    print(f"output_dir={output_dir}")
    print(f"inventory={len(rows)} deployable_gems={len(deployable_gems)} hidden_watchlist={len(hidden_watchlist)}")
    print(f"active_phase2_tag={runtime_ctx.active_phase2_tag} champion_tag={runtime_ctx.champion_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
