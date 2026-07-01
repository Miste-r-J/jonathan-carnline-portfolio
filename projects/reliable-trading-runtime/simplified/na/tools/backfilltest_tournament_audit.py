from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


DEFAULT_BACKFILLTEST_ROOT = Path("backfilltest")
DEFAULT_JSON_OUT = DEFAULT_BACKFILLTEST_ROOT / "backfilltest_audit_ranked.json"
DEFAULT_MD_OUT = DEFAULT_BACKFILLTEST_ROOT / "backfilltest_audit_ranked.md"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class RunArtifacts:
    resolved_config: bool
    run_health_summary: bool
    backfill_diagnostics_report: bool
    backfill_slice_expectancy_report: bool
    removed_trade_impact_report: bool
    state_csv: bool


def classify_run_folder(name: str) -> tuple[str, str, bool]:
    lowered = name.lower()
    smoke_tokens = lowered.startswith("_tmp_smoke") or "_smoke" in lowered or lowered.endswith("smoke")
    if smoke_tokens:
        return "smoke_temp", "smoke_temp", False
    if lowered.startswith("deterministic_fixture") or lowered.startswith("fixture_"):
        return "fixture", "fixture", False
    if lowered.startswith("modelrun20k_candidate"):
        return "candidate_family", "modelrun20k", True
    if lowered.startswith("modelrunlivetests") and lowered[len("modelrunlivetests"):].isdigit():
        return "livetest_family", "livetest", True
    if lowered.startswith("modelrunlivetest") and lowered[len("modelrunlivetest"):].isdigit():
        return "livetest_family", "livetest", True
    if lowered.startswith("modelrun20k_baseline"):
        return "real_run", "modelrun20k", True
    if lowered.startswith("modelrunlivetest"):
        return "unknown", "unknown", True
    if lowered.startswith("modelrun") and lowered[8:].isdigit():
        return "real_run", "modelrun", True
    if lowered.startswith("modelrun") and "_" in lowered:
        return "real_run", "modelrun", True
    return "unknown", "unknown", True


def _artifact_profile(artifacts: RunArtifacts) -> dict[str, Any]:
    artifact_values = asdict(artifacts)
    artifact_count = sum(1 for value in artifact_values.values() if value)
    core_pair_present = artifacts.resolved_config and artifacts.run_health_summary
    core_trio_present = core_pair_present and artifacts.state_csv
    report_pack_count = sum(
        1
        for value in (
            artifacts.backfill_diagnostics_report,
            artifacts.backfill_slice_expectancy_report,
            artifacts.removed_trade_impact_report,
        )
        if value
    )
    report_pack_present = report_pack_count == 3
    report_pack_partial = report_pack_count in {1, 2}
    if artifact_count == 6:
        profile = "full_audit_set"
    elif core_pair_present and artifacts.state_csv and report_pack_count == 0:
        profile = "core_only"
    elif artifacts.resolved_config and artifacts.run_health_summary and artifact_count == 2:
        profile = "fixture_minimal"
    elif artifact_count <= 1:
        profile = "sparse_or_stub"
    else:
        profile = "anomalous_partial"
    return {
        "artifact_count": artifact_count,
        "core_pair_present": core_pair_present,
        "core_trio_present": core_trio_present,
        "report_pack_present": report_pack_present,
        "report_pack_partial": report_pack_partial,
        "missing_core": not core_pair_present,
        "state_without_health": artifacts.state_csv and not artifacts.run_health_summary,
        "state_without_config": artifacts.state_csv and not artifacts.resolved_config,
        "reports_without_state": report_pack_count > 0 and not artifacts.state_csv,
        "empty_folder_like": artifact_count == 0,
        "profile": profile,
    }


def _state_action(row: dict[str, Any]) -> str:
    for key in ("resolved_action", "action", "requested_action"):
        value = str(row.get(key) or "").strip().upper()
        if value:
            return value
    return ""


def reconstruct_state_pnl(state_csv_path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(state_csv_path)
    closed_trades: list[dict[str, Any]] = []
    position = 0
    entry_price: float | None = None
    entry_ts: str | None = None
    ignored_anomalies = Counter()

    for row in rows:
        action = _state_action(row)
        side = str(row.get("side") or "").strip().upper()
        price = _safe_float(row.get("price"))
        ts = str(row.get("datetime") or row.get("ts") or "").strip()
        if action not in {"OPEN", "CLOSE", "FLIP"} or price is None:
            continue
        desired = 1 if side.startswith("L") else -1 if side.startswith("S") else 0
        if action == "OPEN":
            if position == 0 and desired != 0:
                position = desired
                entry_price = price
                entry_ts = ts
            else:
                ignored_anomalies["open_while_active"] += 1
        elif action == "CLOSE":
            if position != 0 and entry_price is not None:
                points = (price - entry_price) if position == 1 else (entry_price - price)
                closed_trades.append(
                    {
                        "entry_ts": entry_ts,
                        "exit_ts": ts,
                        "side": "LONG" if position == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": price,
                        "points": points,
                    }
                )
                position = 0
                entry_price = None
                entry_ts = None
            else:
                ignored_anomalies["close_while_flat"] += 1
        elif action == "FLIP":
            if position != 0 and entry_price is not None:
                points = (price - entry_price) if position == 1 else (entry_price - price)
                closed_trades.append(
                    {
                        "entry_ts": entry_ts,
                        "exit_ts": ts,
                        "side": "LONG" if position == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": price,
                        "points": points,
                    }
                )
            elif position == 0:
                ignored_anomalies["flip_while_flat"] += 1
            if desired != 0:
                position = desired
                entry_price = price
                entry_ts = ts
            else:
                position = 0
                entry_price = None
                entry_ts = None

    total_points = sum(t["points"] for t in closed_trades)
    wins = [t["points"] for t in closed_trades if t["points"] > 0]
    losses = [t["points"] for t in closed_trades if t["points"] < 0]
    win_rate = (len(wins) / len(closed_trades)) if closed_trades else 0.0
    expectancy = (total_points / len(closed_trades)) if closed_trades else 0.0
    best_trade = max((t["points"] for t in closed_trades), default=0.0)
    worst_trade = min((t["points"] for t in closed_trades), default=0.0)
    return {
        "closed_trades": len(closed_trades),
        "wins": len(wins),
        "losses": len(losses),
        "total_points": total_points,
        "usd_1_contract": total_points * 50.0,
        "win_rate": win_rate,
        "expectancy_points": expectancy,
        "best_trade_points": best_trade,
        "worst_trade_points": worst_trade,
        "open_position_at_end": position,
        "open_entry_price": entry_price,
        "open_entry_ts": entry_ts,
        "ignored_anomalies": dict(ignored_anomalies),
    }


def _diagnostics_placeholder(backfill_report: dict[str, Any], expectancy_report: dict[str, Any]) -> tuple[bool, str]:
    expectancy_rows = expectancy_report.get("expectancy_by_slice")
    if not isinstance(expectancy_rows, list) or not expectancy_rows:
        return True, "missing_expectancy_rows"
    trade_total = 0
    all_zero = True
    for row in expectancy_rows:
        trades = int(row.get("trades") or 0)
        trade_total += trades
        metrics = [
            _safe_float(row.get("points_sum")) or 0.0,
            _safe_float(row.get("expectancy_points")) or 0.0,
            _safe_float(row.get("avg_win_points")) or 0.0,
            _safe_float(row.get("avg_loss_points")) or 0.0,
            _safe_float(row.get("win_rate")) or 0.0,
        ]
        if any(abs(x) > 1e-9 for x in metrics):
            all_zero = False
    if trade_total <= 0:
        return True, "zero_trade_expectancy_rows"
    if all_zero:
        return True, "all_zero_expectancy_rows"
    if str(backfill_report.get("expectancy_source") or "").strip() == "trades_csv_fallback" and all_zero:
        return True, "trades_csv_fallback_zeroed"
    return False, ""


def summarize_expectancy(backfill_report: dict[str, Any], expectancy_report: dict[str, Any]) -> dict[str, Any]:
    rows = expectancy_report.get("expectancy_by_slice")
    if not isinstance(rows, list):
        rows = backfill_report.get("expectancy_by_slice")
    if not isinstance(rows, list):
        rows = []
    total_trades = sum(int(row.get("trades") or 0) for row in rows)
    total_points = sum((_safe_float(row.get("points_sum")) or 0.0) for row in rows)
    weighted_win_rate_num = sum((_safe_float(row.get("win_rate")) or 0.0) * int(row.get("trades") or 0) for row in rows)
    weighted_expectancy_num = sum((_safe_float(row.get("expectancy_points")) or 0.0) * int(row.get("trades") or 0) for row in rows)
    sorted_by_points = sorted(rows, key=lambda row: (_safe_float(row.get("points_sum")) or 0.0))
    best_slice = sorted_by_points[-1] if sorted_by_points else {}
    worst_slice = sorted_by_points[0] if sorted_by_points else {}
    return {
        "trades": total_trades,
        "total_points": total_points,
        "usd_1_contract": total_points * 50.0,
        "win_rate": (weighted_win_rate_num / total_trades) if total_trades else 0.0,
        "expectancy_points": (weighted_expectancy_num / total_trades) if total_trades else 0.0,
        "best_slice": best_slice,
        "worst_slice": worst_slice,
    }


def summarize_removed_trade_impact(removed_trade_report: dict[str, Any]) -> dict[str, Any]:
    impact = removed_trade_report.get("removed_trade_impact") if isinstance(removed_trade_report.get("removed_trade_impact"), dict) else {}
    by_reason = impact.get("by_reason") if isinstance(impact.get("by_reason"), dict) else {}
    total_removed = sum(int(v or 0) for v in by_reason.values())
    return {
        "total_removed": total_removed,
        "by_reason": {str(k): int(v or 0) for k, v in by_reason.items()},
    }


def _effective_setting_signature(config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config.get("resolved_thresholds") if isinstance(config.get("resolved_thresholds"), dict) else {}
    entry_gates = config.get("entry_gates") if isinstance(config.get("entry_gates"), dict) else {}
    overlay = config.get("trade_management_overlay") if isinstance(config.get("trade_management_overlay"), dict) else {}
    policy = config.get("phase2_force_open_policy") if isinstance(config.get("phase2_force_open_policy"), dict) else {}
    trade_window = config.get("trade_window_metadata") if isinstance(config.get("trade_window_metadata"), dict) else {}
    regime_policy = config.get("regime_policy") if isinstance(config.get("regime_policy"), dict) else {}
    threshold_resolution = config.get("threshold_resolution") if isinstance(config.get("threshold_resolution"), dict) else {}
    signature = {
        "preset": config.get("preset"),
        "phase2_tag": config.get("phase2_tag"),
        "threshold_active_source": threshold_resolution.get("active_source"),
        "p_setup_required": thresholds.get("p_setup_required"),
        "p_long_required": thresholds.get("p_long_required"),
        "p_short_required": thresholds.get("p_short_required"),
        "threshold_source_setup": (config.get("threshold_sources") or {}).get("p_setup_required") if isinstance(config.get("threshold_sources"), dict) else None,
        "phase2_manifest_thresholds_used": config.get("phase2_manifest_thresholds_used"),
        "gate_vwap": entry_gates.get("gate_vwap"),
        "gate_ema": entry_gates.get("gate_ema"),
        "gate_tod": entry_gates.get("gate_tod"),
        "vwap_gate_mode": entry_gates.get("vwap_gate_mode"),
        "gate_mode": entry_gates.get("gate_mode"),
        "effective_trade_start": trade_window.get("effective_start"),
        "effective_trade_end": trade_window.get("effective_end"),
        "effective_trade_reason": trade_window.get("effective_reason"),
        "phase2_force_open_enabled": policy.get("enabled"),
        "phase2_force_open_min_setup": policy.get("min_setup"),
        "phase2_force_open_min_entry_conf": policy.get("min_entry_conf"),
        "phase2_force_open_live_only": policy.get("live_only"),
        "pnl_overlay_enabled": overlay.get("pnl_overlay_enabled"),
        "pnl_giveback_activate_r": overlay.get("pnl_giveback_activate_r"),
        "pnl_giveback_close_r": overlay.get("pnl_giveback_close_r"),
        "pnl_runner_enabled": overlay.get("pnl_runner_enabled"),
        "pnl_runner_arm_r": overlay.get("pnl_runner_arm_r"),
        "pnl_runner_giveback_r": overlay.get("pnl_runner_giveback_r"),
        "pnl_runner_suppress_target": overlay.get("pnl_runner_suppress_target"),
        "pnl_shelf_enabled": config.get("pnl_shelf_enabled"),
        "allow_countertrend_in_unresolved": regime_policy.get("allow_countertrend_in_unresolved"),
        "allow_countertrend_fade_in_trend": regime_policy.get("allow_countertrend_fade_in_trend"),
    }
    return signature


def _signature_id(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _run_setting_cluster_key(run: dict[str, Any]) -> str:
    return str(run.get("signature_id") or "")


def _sort_key_for_competitive(run: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(run.get("balanced_edge_score") or 0.0),
        float(run.get("expectancy_points") or 0.0),
        float(run.get("worst_slice_points") or 0.0),
        float(run.get("trade_count") or 0.0),
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _minmax_norm(value: float, values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return 1.0 if hi > 0 else 0.0
    return (value - lo) / (hi - lo)


def _detect_distortions(run_dir: Path, config: dict[str, Any], health: dict[str, Any], placeholder_reason: str, fallback_used: bool, backfill_report: dict[str, Any]) -> dict[str, Any]:
    trade_window = config.get("trade_window_metadata") if isinstance(config.get("trade_window_metadata"), dict) else {}
    threshold_resolution = config.get("threshold_resolution") if isinstance(config.get("threshold_resolution"), dict) else {}
    distortions = {
        "tod_disabled_24h_drift": False,
        "preset_manifest_threshold_divergence": False,
        "placeholder_zeroed_diagnostics": bool(placeholder_reason),
        "sparse_trades_csv_fallback": False,
        "state_only_profitability_without_execution_evidence": False,
        "artifact_incomplete": False,
    }
    if str(trade_window.get("effective_reason") or "") == "gate_tod_disabled_auto_24h":
        distortions["tod_disabled_24h_drift"] = True
    if threshold_resolution.get("preset_manifest_divergence") or threshold_resolution.get("preset_manifest_mismatch"):
        distortions["preset_manifest_threshold_divergence"] = True
    if str(backfill_report.get("expectancy_source") or "") == "trades_csv_fallback":
        distortions["sparse_trades_csv_fallback"] = True
    if fallback_used and int(((health.get("trade_evidence") or {}).get("executable_fill_rows") or 0) == 0):
        distortions["state_only_profitability_without_execution_evidence"] = True
    must_have = ["resolved_config.json", "run_health_summary.json", "state.csv"]
    distortions["artifact_incomplete"] = any(not (run_dir / name).exists() for name in must_have)
    return distortions


def build_run_record(run_dir: Path) -> dict[str, Any]:
    classification, family, competitive = classify_run_folder(run_dir.name)
    resolved_config = _read_json(run_dir / "resolved_config.json")
    run_health = _read_json(run_dir / "run_health_summary.json")
    backfill_report = _read_json(run_dir / "backfill_diagnostics_report.json")
    expectancy_report = _read_json(run_dir / "backfill_slice_expectancy_report.json")
    removed_trade_report = _read_json(run_dir / "removed_trade_impact_report.json")
    state_csv_path = run_dir / "state.csv"

    placeholder, placeholder_reason = _diagnostics_placeholder(backfill_report, expectancy_report)
    expectancy_summary = summarize_expectancy(backfill_report, expectancy_report)
    fallback_summary = reconstruct_state_pnl(state_csv_path) if state_csv_path.exists() else {}
    fallback_used = placeholder or expectancy_summary["trades"] <= 0
    metric_source = "state_csv_fallback" if fallback_used else "diagnostics_reports"
    active_summary = fallback_summary if fallback_used else expectancy_summary
    removed_summary = summarize_removed_trade_impact(removed_trade_report)
    phase_counters = run_health.get("executor_stats_by_phase") if isinstance(run_health.get("executor_stats_by_phase"), dict) else {}
    distortions = _detect_distortions(
        run_dir=run_dir,
        config=resolved_config,
        health=run_health,
        placeholder_reason=placeholder_reason,
        fallback_used=fallback_used,
        backfill_report=backfill_report,
    )
    artifacts = RunArtifacts(
        resolved_config=(run_dir / "resolved_config.json").exists(),
        run_health_summary=(run_dir / "run_health_summary.json").exists(),
        backfill_diagnostics_report=(run_dir / "backfill_diagnostics_report.json").exists(),
        backfill_slice_expectancy_report=(run_dir / "backfill_slice_expectancy_report.json").exists(),
        removed_trade_impact_report=(run_dir / "removed_trade_impact_report.json").exists(),
        state_csv=state_csv_path.exists(),
    )
    artifact_profile = _artifact_profile(artifacts)
    signature = _effective_setting_signature(resolved_config)
    signature_id = _signature_id(signature)
    best_slice = active_summary.get("best_slice") if isinstance(active_summary.get("best_slice"), dict) else {}
    worst_slice = active_summary.get("worst_slice") if isinstance(active_summary.get("worst_slice"), dict) else {}
    trade_evidence = run_health.get("trade_evidence") if isinstance(run_health.get("trade_evidence"), dict) else {}
    executor_stats = run_health.get("executor_stats_all_phases") if isinstance(run_health.get("executor_stats_all_phases"), dict) else {}
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "classification": classification,
        "family": family,
        "competitive": competitive,
        "artifacts": asdict(artifacts),
        "artifact_profile": artifact_profile,
        "metric_source": metric_source,
        "generated_at": backfill_report.get("generated_at") or expectancy_report.get("generated_at"),
        "run_id": backfill_report.get("run_id") or expectancy_report.get("run_id") or run_health.get("run_id"),
        "placeholder_reason": placeholder_reason or None,
        "preset": resolved_config.get("preset"),
        "phase2_tag": resolved_config.get("phase2_tag"),
        "expectancy_source": backfill_report.get("expectancy_source"),
        "source_files": backfill_report.get("source_files"),
        "resolved_config": {
            "threshold_resolution": resolved_config.get("threshold_resolution"),
            "trade_window_metadata": resolved_config.get("trade_window_metadata"),
            "entry_gates": resolved_config.get("entry_gates"),
            "trade_management_overlay": resolved_config.get("trade_management_overlay"),
            "phase2_force_open_policy": resolved_config.get("phase2_force_open_policy"),
            "regime_policy": resolved_config.get("regime_policy"),
        },
        "signature": signature,
        "signature_id": signature_id,
        "thresholds": {
            "p_setup_required": signature.get("p_setup_required"),
            "p_long_required": signature.get("p_long_required"),
            "p_short_required": signature.get("p_short_required"),
            "source": signature.get("threshold_source_setup"),
        },
        "trade_window": resolved_config.get("trade_window_metadata"),
        "gate_settings": resolved_config.get("entry_gates"),
        "overlay_settings": resolved_config.get("trade_management_overlay"),
        "force_open_policy": resolved_config.get("phase2_force_open_policy"),
        "trade_count": int(active_summary.get("trades") or active_summary.get("closed_trades") or 0),
        "total_points": float(active_summary.get("total_points") or 0.0),
        "usd_1_contract": float(active_summary.get("usd_1_contract") or 0.0),
        "win_rate": float(active_summary.get("win_rate") or 0.0),
        "expectancy_points": float(active_summary.get("expectancy_points") or 0.0),
        "best_slice_points": float((_safe_float(best_slice.get("points_sum")) or active_summary.get("best_trade_points") or 0.0)),
        "worst_slice_points": float((_safe_float(worst_slice.get("points_sum")) or active_summary.get("worst_trade_points") or 0.0)),
        "best_slice": best_slice or None,
        "worst_slice": worst_slice or None,
        "removed_trade_impact": removed_summary,
        "run_verdict": run_health.get("verdict"),
        "insufficient_live_evidence": run_health.get("insufficient_live_evidence"),
        "execution_evidence": {
            "executable_fill_rows": int(trade_evidence.get("executable_fill_rows") or 0),
            "total_rows": int(trade_evidence.get("total_rows") or 0),
            "executor_sent_total": int(executor_stats.get("executor_sent_total") or 0),
            "nt_order_entry_total": int(executor_stats.get("nt_order_entry_total") or 0),
        },
        "phase_counters": phase_counters,
        "raw_optional_health_json": run_health,
        "distortions": distortions,
        "fallback_summary": fallback_summary if fallback_summary else None,
    }


def _assign_balanced_scores(runs: list[dict[str, Any]]) -> None:
    competitive_runs = [run for run in runs if run.get("competitive")]
    points_values = [float(run.get("total_points") or 0.0) for run in competitive_runs]
    expectancy_values = [float(run.get("expectancy_points") or 0.0) for run in competitive_runs]
    win_rate_values = [float(run.get("win_rate") or 0.0) for run in competitive_runs]
    worst_slice_severity = [abs(min(0.0, float(run.get("worst_slice_points") or 0.0))) for run in competitive_runs]
    removed_values = [int(((run.get("removed_trade_impact") or {}).get("total_removed") or 0)) for run in competitive_runs]
    trade_counts = [int(run.get("trade_count") or 0) for run in competitive_runs]
    for run in runs:
        points = float(run.get("total_points") or 0.0)
        expectancy = float(run.get("expectancy_points") or 0.0)
        win_rate = float(run.get("win_rate") or 0.0)
        trade_count = int(run.get("trade_count") or 0)
        worst_abs = abs(min(0.0, float(run.get("worst_slice_points") or 0.0)))
        removed_total = int(((run.get("removed_trade_impact") or {}).get("total_removed") or 0))
        points_norm = _minmax_norm(points, points_values)
        expectancy_norm = _minmax_norm(expectancy, expectancy_values)
        win_rate_norm = _minmax_norm(win_rate, win_rate_values)
        trade_sufficiency = _clamp01(trade_count / 40.0)
        worst_slice_norm = 1.0 - _minmax_norm(worst_abs, worst_slice_severity)
        removed_norm = 1.0 - _minmax_norm(removed_total, removed_values)
        raw_score = (
            0.35 * points_norm
            + 0.25 * expectancy_norm
            + 0.15 * win_rate_norm
            + 0.10 * trade_sufficiency
            + 0.10 * worst_slice_norm
            + 0.05 * removed_norm
        ) * 100.0
        penalties = 0.0
        distortions = run.get("distortions") or {}
        if distortions.get("tod_disabled_24h_drift"):
            penalties += 12.0
        if distortions.get("preset_manifest_threshold_divergence"):
            penalties += 8.0
        if distortions.get("placeholder_zeroed_diagnostics"):
            penalties += 12.0
        if distortions.get("sparse_trades_csv_fallback"):
            penalties += 6.0
        if distortions.get("state_only_profitability_without_execution_evidence"):
            penalties += 6.0
        if distortions.get("artifact_incomplete"):
            penalties += 8.0
        if trade_count < 5:
            penalties += 15.0
        elif trade_count < 15:
            penalties += 8.0
        if not run.get("competitive"):
            penalties += 100.0
        run["balanced_edge_score"] = round(raw_score - penalties, 3)


def _rank_runs(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full = sorted(runs, key=_sort_key_for_competitive, reverse=True)
    competitive = sorted([run for run in runs if run.get("competitive")], key=_sort_key_for_competitive, reverse=True)
    for idx, run in enumerate(full, start=1):
        run["full_corpus_rank"] = idx
    for idx, run in enumerate(competitive, start=1):
        run["competitive_rank"] = idx
    return full, competitive


def _signature_distance(a: dict[str, Any], b: dict[str, Any]) -> int:
    keys = sorted(set(a.keys()) | set(b.keys()))
    return sum(1 for key in keys if a.get(key) != b.get(key))


def _winner_summary(competitive_rankings: list[dict[str, Any]], all_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not competitive_rankings:
        return {}
    winner = competitive_rankings[0]
    siblings = []
    for candidate in all_runs:
        if candidate["run_name"] == winner["run_name"]:
            continue
        distance = _signature_distance(winner.get("signature") or {}, candidate.get("signature") or {})
        siblings.append(
            {
                "run_name": candidate["run_name"],
                "preset": candidate.get("preset"),
                "balanced_edge_score": candidate.get("balanced_edge_score"),
                "competitive": candidate.get("competitive"),
                "distance": distance,
            }
        )
    siblings.sort(key=lambda row: (row["distance"], -float(row["balanced_edge_score"] or 0.0)))
    top_siblings = siblings[:3]

    same_signature = [
        run for run in all_runs
        if run.get("signature_id") == winner.get("signature_id") and run["run_name"] != winner["run_name"]
    ]
    same_signature.sort(key=_sort_key_for_competitive, reverse=True)

    deltas = []
    if top_siblings:
        first = next((run for run in all_runs if run["run_name"] == top_siblings[0]["run_name"]), None)
        if first is not None:
            for key in sorted(set(winner["signature"].keys()) | set(first["signature"].keys())):
                if winner["signature"].get(key) != first["signature"].get(key):
                    deltas.append(
                        {
                            "field": key,
                            "winner": winner["signature"].get(key),
                            "peer": first["signature"].get(key),
                        }
                    )

    repeated_hurts = Counter()
    for run in competitive_rankings[-10:]:
        for key, enabled in (run.get("distortions") or {}).items():
            if enabled:
                repeated_hurts[key] += 1
    hurting_settings = [{"signal": key, "count": count} for key, count in repeated_hurts.most_common(8)]

    return {
        "winning_run": winner["run_name"],
        "winning_preset": winner.get("preset"),
        "winning_score": winner.get("balanced_edge_score"),
        "winning_signature_id": winner.get("signature_id"),
        "winning_signature": winner.get("signature"),
        "top_3_nearest_siblings": top_siblings,
        "same_signature_runs": [run["run_name"] for run in same_signature[:5]],
        "specific_deltas_vs_nearest_peer": deltas[:12],
        "settings_that_repeatedly_hurt_performance": hurting_settings,
    }


def _cluster_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[_run_setting_cluster_key(run)].append(run)
    output = []
    for signature_id, grouped_runs in groups.items():
        grouped_runs.sort(key=_sort_key_for_competitive, reverse=True)
        scores = [float(run.get("balanced_edge_score") or 0.0) for run in grouped_runs]
        competitive_runs = [run for run in grouped_runs if run.get("competitive")]
        competitive_scores = [float(run.get("balanced_edge_score") or 0.0) for run in competitive_runs]
        total_trades = sum(int(run.get("trade_count") or 0) for run in competitive_runs)
        zero_trade_runs = sum(1 for run in competitive_runs if int(run.get("trade_count") or 0) <= 2)
        confidence = min(1.0, math.sqrt(max(total_trades, 0) / 50.0)) * min(1.0, math.sqrt(max(len(competitive_runs), 0) / 5.0)) if competitive_runs else 0.0
        cleanliness = 1.0
        if competitive_runs:
            if any((run.get("distortions") or {}).get("tod_disabled_24h_drift") for run in competitive_runs):
                cleanliness *= 0.70
            if any(str((run.get("signature") or {}).get("threshold_active_source") or "").lower() == "mixed" for run in competitive_runs):
                cleanliness *= 0.80
            if (zero_trade_runs / max(len(competitive_runs), 1)) > 0.5:
                cleanliness *= 0.75
        family_median_edge = median(competitive_scores) if competitive_scores else 0.0
        family_score = family_median_edge * confidence * cleanliness
        output.append(
            {
                "signature_id": signature_id,
                "signature": grouped_runs[0].get("signature"),
                "preset": grouped_runs[0].get("preset"),
                "run_count": len(grouped_runs),
                "competitive_run_count": sum(1 for run in grouped_runs if run.get("competitive")),
                "avg_score": round(mean(scores), 3) if scores else 0.0,
                "median_competitive_score": round(family_median_edge, 3) if competitive_scores else 0.0,
                "family_confidence": round(confidence, 3),
                "family_cleanliness": round(cleanliness, 3),
                "family_score": round(family_score, 3),
                "best_run": grouped_runs[0]["run_name"],
                "best_score": grouped_runs[0].get("balanced_edge_score"),
                "total_competitive_trades": total_trades,
                "zero_trade_or_sparse_run_count": zero_trade_runs,
                "runs": [run["run_name"] for run in grouped_runs[:10]],
            }
        )
    output.sort(key=lambda row: (row["family_score"], row["competitive_run_count"], row["avg_score"]), reverse=True)
    return output


def _best_setting_summary(clusters: list[dict[str, Any]], all_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not clusters:
        return {}
    winner_cluster = next((cluster for cluster in clusters if int(cluster.get("competitive_run_count") or 0) > 0), None)
    if winner_cluster is None:
        return {}
    signature_id = str(winner_cluster.get("signature_id") or "")
    family_runs = [run for run in all_runs if run.get("signature_id") == signature_id]
    family_runs.sort(key=_sort_key_for_competitive, reverse=True)
    same_signature_runs = [run["run_name"] for run in family_runs[1:6]]
    top_run = family_runs[0] if family_runs else {}
    return {
        "winning_signature_id": signature_id,
        "winning_signature": winner_cluster.get("signature"),
        "winning_preset": winner_cluster.get("preset"),
        "winning_family_score": winner_cluster.get("family_score"),
        "winning_run": top_run.get("run_name"),
        "winning_run_score": top_run.get("balanced_edge_score"),
        "run_count": winner_cluster.get("run_count"),
        "competitive_run_count": winner_cluster.get("competitive_run_count"),
        "total_competitive_trades": winner_cluster.get("total_competitive_trades"),
        "same_signature_runs": same_signature_runs,
    }


def audit_backfilltest(root: Path) -> dict[str, Any]:
    run_dirs = [path for path in sorted(root.iterdir(), key=lambda p: p.name.lower()) if path.is_dir()]
    run_records = [build_run_record(path) for path in run_dirs]
    _assign_balanced_scores(run_records)
    full_rankings, competitive_rankings = _rank_runs(run_records)
    clusters = _cluster_summary(run_records)
    winner = _winner_summary(competitive_rankings, run_records)
    best_setting = _best_setting_summary(clusters, run_records)
    summary = {
        "run_count_total": len(run_records),
        "competitive_run_count": sum(1 for run in run_records if run.get("competitive")),
        "classification_counts": dict(Counter(run["classification"] for run in run_records)),
        "metric_source_counts": dict(Counter(run["metric_source"] for run in run_records)),
        "preset_counts": dict(Counter(str(run.get("preset") or "UNKNOWN") for run in run_records)),
    }
    return {
        "root": str(root),
        "summary": summary,
        "winner": winner,
        "best_setting": best_setting,
        "setting_clusters": clusters,
        "full_corpus_rankings": full_rankings,
        "competitive_rankings": competitive_rankings,
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    winner = payload.get("winner") or {}
    best_setting = payload.get("best_setting") or {}
    competitive = payload.get("competitive_rankings") or []
    full = payload.get("full_corpus_rankings") or []
    lines = [
        "# Backfilltest Audit Tournament",
        "",
        f"- Root: `{payload.get('root')}`",
        f"- Total folders audited: `{summary.get('run_count_total', 0)}`",
        f"- Competitive runs: `{summary.get('competitive_run_count', 0)}`",
        "",
        "## Winner",
        "",
        f"- Winning run: `{winner.get('winning_run', 'n/a')}`",
        f"- Preset: `{winner.get('winning_preset', 'n/a')}`",
        f"- Balanced edge score: `{winner.get('winning_score', 'n/a')}`",
        f"- Signature id: `{winner.get('winning_signature_id', 'n/a')}`",
        "",
        "## Best Setting Family",
        "",
        f"- Best setting preset: `{best_setting.get('winning_preset', 'n/a')}`",
        f"- Family score: `{best_setting.get('winning_family_score', 'n/a')}`",
        f"- Lead run: `{best_setting.get('winning_run', 'n/a')}`",
        f"- Competitive runs in family: `{best_setting.get('competitive_run_count', 'n/a')}`",
        f"- Total competitive trades in family: `{best_setting.get('total_competitive_trades', 'n/a')}`",
        "",
        "## Top Competitive Runs",
        "",
    ]
    for run in competitive[:10]:
        lines.append(
            f"- `{run['competitive_rank']}`. `{run['run_name']}` | preset `{run.get('preset')}` | "
            f"score `{run.get('balanced_edge_score')}` | points `{round(float(run.get('total_points') or 0.0), 2)}` | "
            f"expectancy `{round(float(run.get('expectancy_points') or 0.0), 3)}` | trades `{run.get('trade_count')}`"
        )
    lines.extend(["", "## Winner Siblings", ""])
    for row in winner.get("top_3_nearest_siblings") or []:
        lines.append(
            f"- `{row['run_name']}` | distance `{row['distance']}` | score `{row['balanced_edge_score']}` | competitive `{row['competitive']}`"
        )
    lines.extend(["", "## Repeated Hurts", ""])
    for row in winner.get("settings_that_repeatedly_hurt_performance") or []:
        lines.append(f"- `{row['signal']}` appeared in `{row['count']}` low-ranked competitive runs")
    lines.extend(["", "## Full-Corpus Notes", ""])
    for run in full[:10]:
        lines.append(
            f"- `{run['full_corpus_rank']}`. `{run['run_name']}` | class `{run['classification']}` | "
            f"competitive `{run['competitive']}` | score `{run.get('balanced_edge_score')}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit backfilltest runs and rank the best effective setting by balanced edge.")
    parser.add_argument("--root", default=str(DEFAULT_BACKFILLTEST_ROOT), help="Backfilltest root directory.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="Output JSON report path.")
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT), help="Output markdown summary path.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = audit_backfilltest(root)
    json_out = Path(args.json_out).resolve()
    md_out = Path(args.md_out).resolve()
    json_out.write_text(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2), encoding="utf-8")
    md_out.write_text(build_markdown_report(payload), encoding="utf-8")
    print(str(json_out))
    print(str(md_out))


if __name__ == "__main__":
    main()
