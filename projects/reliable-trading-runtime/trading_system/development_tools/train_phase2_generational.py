from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_sharp import run_candidate  # type: ignore


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a 5-generation Phase-2 training ladder with hard promotion gates.")
    p.add_argument("--csv", default=str(REPO_ROOT / "data" / "intraday" / "es" / "ES6.csv"))
    p.add_argument("--baseline-tag", default="retrain_v4")
    p.add_argument("--deployment-baseline-tag", default="retrain_v6_pass2_grid_02")
    p.add_argument("--artifact-root", default=str(ROOT / "artifacts" / "phase2" / "candidates"))
    p.add_argument("--summary-dir", default=str(ROOT / "runs" / "phase2_generational"))
    p.add_argument("--tag-prefix", default="retrain_v7")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--generation-budget", type=int, default=8)
    p.add_argument("--search-bias", choices=["auto", "trade_recovery", "fade_control", "balanced"], default="auto")
    p.add_argument("--promote-on", choices=["risk_adjusted_oos"], default="risk_adjusted_oos")
    p.add_argument("--drawdown-tolerance", type=float, default=1.05)
    p.add_argument("--loss-streak-tolerance", type=int, default=0)
    p.add_argument("--max-bad-fade-rate", type=float, default=0.18)
    p.add_argument("--max-flip-rate", type=float, default=0.75)
    p.add_argument("--min-trades-val", type=int, default=30)
    p.add_argument("--require-slippage-pass", dest="require_slippage_pass", action="store_true", default=True)
    p.add_argument("--allow-slippage-fail", dest="require_slippage_pass", action="store_false")
    p.add_argument("--walkforward-windows", type=int, default=3)
    p.add_argument("--live-shadow-summary-path", default="run_health_summary.json")
    p.add_argument("--require-safe-shadow-pass", dest="require_safe_shadow_pass", action="store_true", default=True)
    p.add_argument("--allow-unsafe-shadow-pass", dest="require_safe_shadow_pass", action="store_false")
    p.add_argument("--min-trades-floor", type=int, default=20)
    p.add_argument("--train-start", default="2021-01-14")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--val-start", default="2025-06-01")
    p.add_argument("--val-end", default="2025-10-31")
    p.add_argument("--test-start", default="2025-11-01")
    p.add_argument("--test-end", default="2026-01-16")
    p.add_argument("--n-estimators", type=int, default=1600)
    p.add_argument("--benchmark-after-generation", action="store_true", help="Run the advisory Phase-2 benchmark report after each generation.")
    p.add_argument("--execute", action="store_true", help="Actually run training/validation/evaluation commands.")
    p.add_argument("--fresh-run", action="store_true", help="Ignore prior summary_dir lineage state and start a fresh run from the deployment baseline.")
    return p.parse_args()


def _run(cmd: List[str], *, execute: bool, cwd: Path = ROOT) -> subprocess.CompletedProcess[str] | None:
    print("\n$ " + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    if not execute:
        return None
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc


def _load_manifest(tag: str, artifact_root: Path) -> Dict[str, Any]:
    path = artifact_root / tag / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found for tag {tag}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_manifest_path"] = str(path)
    payload["_manifest_dir"] = str(path.parent)
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _safe_run(cmd: List[str], *, execute: bool, cwd: Path = ROOT) -> tuple[bool, subprocess.CompletedProcess[str] | None, Optional[str]]:
    try:
        return True, _run(cmd, execute=execute, cwd=cwd), None
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        return False, None, detail or f"command_failed_returncode={exc.returncode}"
    except Exception as exc:
        return False, None, str(exc)


def _max_consecutive_losses(trades: List[Dict[str, Any]]) -> int:
    streak = 0
    max_streak = 0
    for trade in trades:
        pnl = float(trade.get("pnl_usd") or 0.0)
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _monthly_pnl(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    buckets: Dict[str, float] = {}
    for trade in trades:
        ts = trade.get("entry_ts")
        dt = None
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            dt = None
        if dt is None:
            continue
        key = dt.strftime("%Y-%m")
        buckets[key] = float(buckets.get(key) or 0.0) + float(trade.get("pnl_usd") or 0.0)
    return dict(sorted(buckets.items()))


def _seed_champion(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "tag": str(args.deployment_baseline_tag or args.baseline_tag),
        "config": {
            "label_mode": "horizon",
            "horizon": 8,
            "label_threshold": 0.0015,
            "threshold_objective": "sharpe",
            "stack_setup_prob": False,
            "mtf_timeframes": [],
            "walkforward_windows": int(args.walkforward_windows),
            "setup_threshold_multiplier": 1.75,
            "max_bad_fade_rate": args.max_bad_fade_rate,
            "close_replay": {
                "pnl_giveback_close_r": 0.5,
                "pnl_stall_bars": 4,
                "pnl_severe_adverse_r": 0.9,
            },
        },
        "setup_selectivity_stats": {
            "trade_rate": 0.18,
            "trade_threshold": 0.35,
            "precision_trade_test": 0.5,
            "recall_trade_test": 0.3,
        },
        "behavior_audit_test": {
            "countertrend_rate": 0.12,
        },
        "slippage": {
            "slip_1": {
                "profit_factor": 1.0,
                "sharpe": 0.0,
                "max_drawdown": 1000.0,
                "trade_count": args.min_trades_floor,
            }
        },
        "max_consecutive_losses": 4,
        "risk_score": 0.0,
        "promoted": True,
    }


def _setup_selectivity(manifest: Dict[str, Any]) -> Dict[str, Any]:
    stats = dict(manifest.get("setup_selectivity_stats") or {})
    if stats:
        return stats
    setup = ((manifest.get("metrics") or {}).get("setup") or {}) if isinstance(manifest.get("metrics"), dict) else {}
    class_counts = ((setup.get("label_info") or {}).get("class_counts") or {}) if isinstance(setup, dict) else {}
    trade = int(class_counts.get("trade") or 0)
    flat = int(class_counts.get("flat") or 0)
    total = max(trade + flat, 1)
    return {
        "trade_labels": trade,
        "flat_labels": flat,
        "trade_rate": float(trade) / float(total),
        "trade_threshold": float(setup.get("trade_threshold") or 0.0),
        "precision_trade_test": float(setup.get("precision_trade_test") or 0.0),
        "recall_trade_test": float(setup.get("recall_trade_test") or 0.0),
    }


def _behavior_audit(manifest: Dict[str, Any]) -> Dict[str, Any]:
    audit = manifest.get("behavior_audit_test") or manifest.get("behavior_audit_val") or {}
    return dict(audit) if isinstance(audit, dict) else {}


def _sharp_args(tag: str, csv_path: str, slippage: float) -> SimpleNamespace:
    return SimpleNamespace(
        tag=tag,
        manifest=None,
        csv=csv_path,
        instrument="ES",
        contracts=1,
        trade_window_start="00:00",
        trade_window_end="23:59",
        max_hold_bars=24,
        out_dir=str(ROOT / "runs" / "phase2_generational" / "_sharp_tmp"),
        p_setup=None,
        p_long=None,
        p_short=None,
        start_at=None,
        end_at=None,
        commission_per_contract=2.0,
        slippage_ticks=slippage,
        skip_store=True,
    )


def _rejected_candidate_row(manifest: Dict[str, Any], *, generation: int, parent_tag: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tag": manifest.get("tag"),
        "manifest_path": manifest.get("_manifest_path"),
        "thresholds": manifest.get("thresholds") or {},
        "threshold_diagnostics": manifest.get("threshold_diagnostics") or {},
        "config": manifest.get("config") or {},
        "behavior_audit_test": manifest.get("behavior_audit_test") or {},
        "setup_selectivity_stats": _setup_selectivity(manifest),
        "monthly_stability": {},
        "slippage": {},
        "max_consecutive_losses": 0,
        "slippage_pass": False,
        "risk_score": float("-inf"),
        "rejected": True,
        "rejected_reason": manifest.get("rejected_reason"),
        "generation": generation,
        "parent_tag": parent_tag,
        "candidate_config": cfg,
        "promotion_pass": False,
        "promotion_reasons": [f"trainer rejected candidate: {manifest.get('rejected_reason') or 'unknown'}"],
        "candidate_status": str(manifest.get("candidate_status") or "rejected"),
    }


def _failed_candidate_row(
    *,
    tag: str,
    artifact_root: Path,
    generation: int,
    parent_tag: str,
    cfg: Dict[str, Any],
    reason: str,
    status: str,
    thresholds: Optional[Dict[str, Any]] = None,
    threshold_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest_path = artifact_root / tag / "manifest.json"
    return {
        "tag": tag,
        "manifest_path": str(manifest_path),
        "thresholds": thresholds or {},
        "threshold_diagnostics": threshold_diagnostics or {},
        "config": {},
        "behavior_audit_test": {},
        "setup_selectivity_stats": {},
        "monthly_stability": {},
        "slippage": {},
        "max_consecutive_losses": 0,
        "slippage_pass": False,
        "risk_score": float("-inf"),
        "rejected": True,
        "rejected_reason": reason,
        "generation": generation,
        "parent_tag": parent_tag,
        "candidate_config": cfg,
        "promotion_pass": False,
        "promotion_reasons": [reason],
        "candidate_status": status,
    }

def _evaluate_tag(args: argparse.Namespace, artifact_root: Path, tag: str) -> Dict[str, Any]:
    manifest = _load_manifest(tag, artifact_root)
    slippage_results: Dict[str, Dict[str, Any]] = {}
    trades_for_loss: List[Dict[str, Any]] = []
    for slippage in (0.0, 1.0, 2.0):
        sharp_args = _sharp_args(tag, args.csv, slippage)
        sharp_args.start_at = args.test_start
        sharp_args.end_at = args.test_end
        result = run_candidate(sharp_args)
        sim = result.get("sim") or {}
        slippage_results[f"slip_{slippage:g}"] = {
            "total_pnl_usd": float(sim.get("total_pnl_usd") or 0.0),
            "profit_factor": float(sim.get("profit_factor") or 0.0),
            "max_drawdown": float(sim.get("max_drawdown") or 0.0),
            "sharpe": float(sim.get("sharpe") or 0.0),
            "trade_count": int(sim.get("trade_count") or len(sim.get("trades") or [])),
            "trades_per_day": float(sim.get("trades_per_day") or 0.0),
            "flip_rate_per_day": float(sim.get("flip_rate_per_day") or 0.0),
            "win_rate": float(sim.get("win_rate") or 0.0),
        }
        if abs(slippage - 1.0) < 1e-9:
            trades_for_loss = list(sim.get("trades") or [])
    slip1 = slippage_results["slip_1"]
    behavior = _behavior_audit(manifest)
    selectivity = _setup_selectivity(manifest)
    max_loss_streak = _max_consecutive_losses(trades_for_loss)
    monthly = _monthly_pnl(trades_for_loss)
    slippage_pass = all(
        (payload.get("trade_count") or 0) >= max(1, args.min_trades_floor // 2)
        and float(payload.get("profit_factor") or 0.0) >= 1.0
        and float(payload.get("total_pnl_usd") or 0.0) > 0.0
        for payload in slippage_results.values()
    )
    bad_fade_rate = float(behavior.get("countertrend_rate") or 0.0)
    risk_score = (
        float(slip1.get("sharpe") or 0.0) * 100.0
        + float(slip1.get("profit_factor") or 0.0) * 50.0
        + float(slip1.get("total_pnl_usd") or 0.0) / 100.0
        - abs(float(slip1.get("max_drawdown") or 0.0)) / 100.0
        - float(max_loss_streak) * 10.0
        - bad_fade_rate * 100.0
    )
    live_shadow_gate = manifest.get("live_shadow_gate") or {}
    return {
        "tag": tag,
        "manifest_path": manifest.get("_manifest_path"),
        "thresholds": manifest.get("thresholds") or {},
        "threshold_diagnostics": manifest.get("threshold_diagnostics") or {},
        "config": manifest.get("config") or {},
        "behavior_audit_test": behavior,
        "setup_selectivity_stats": selectivity,
        "monthly_stability": monthly,
        "slippage": slippage_results,
        "max_consecutive_losses": max_loss_streak,
        "slippage_pass": slippage_pass,
        "risk_score": risk_score,
        "walkforward": manifest.get("walkforward") or {},
        "live_shadow_gate": live_shadow_gate,
        "live_shadow_pass": bool(live_shadow_gate.get("passed")),
        "promotion_blocked": bool(manifest.get("promotion_blocked")),
        "promotion_blocked_reason": manifest.get("promotion_blocked_reason"),
        "rejected": bool(manifest.get("rejected")),
        "rejected_reason": manifest.get("rejected_reason"),
        "candidate_status": str(manifest.get("candidate_status") or "evaluated"),
    }


def _update_candidate_manifest(manifest_path: Path, updates: Dict[str, Any]) -> None:
    payload = _read_json(manifest_path, {})
    payload.update(updates)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_manifest_path(artifact_root: Path, tag: str) -> Path:
    return artifact_root / tag / "manifest.json"


def _set_candidate_status(
    artifact_root: Path,
    tag: str,
    *,
    generation: int,
    parent_tag: str,
    promotion_target_tag: str,
    status: str,
    promotion_result: str,
    extra_updates: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest_path = _candidate_manifest_path(artifact_root, tag)
    payload = _read_json(manifest_path, {})
    payload.update(
        {
            "tag": payload.get("tag") or tag,
            "generation": generation,
            "parent_tag": parent_tag,
            "promotion_target_tag": promotion_target_tag,
            "promotion_result": promotion_result,
            "candidate_status": status,
            "status_updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if extra_updates:
        payload.update(extra_updates)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _record_candidate_failure(
    artifact_root: Path,
    tag: str,
    *,
    generation: int,
    parent_tag: str,
    reason: str,
    status: str,
    thresholds: Optional[Dict[str, Any]] = None,
    threshold_diagnostics: Optional[Dict[str, Any]] = None,
) -> Path:
    return _set_candidate_status(
        artifact_root,
        tag,
        generation=generation,
        parent_tag=parent_tag,
        promotion_target_tag=parent_tag,
        status=status,
        promotion_result="failed",
        extra_updates={
            "rejected": True,
            "rejected_reason": reason,
            "thresholds": thresholds or {},
            "threshold_diagnostics": threshold_diagnostics or {},
        },
    )


def _train_cmd(args: argparse.Namespace, tag: str, cfg: Dict[str, Any], generation: int, parent_tag: str) -> List[str]:
    cmd = [
        sys.executable,
        "tools/train_phase2_grid.py",
        "--csv",
        str(args.csv),
        "--instrument",
        "ES",
        "--tag",
        tag,
        "--artifact-root",
        str(args.artifact_root),
        "--train-start",
        args.train_start,
        "--train-end",
        args.train_end,
        "--val-start",
        args.val_start,
        "--val-end",
        args.val_end,
        "--test-start",
        args.test_start,
        "--test-end",
        args.test_end,
        "--label-mode",
        str(cfg["label_mode"]),
        "--horizon",
        str(cfg["horizon"]),
        "--label-threshold",
        str(cfg["label_threshold"]),
        "--label-max-hold-bars",
        str(cfg.get("label_max_hold_bars", cfg["horizon"])),
        "--threshold-objective",
        str(cfg["threshold_objective"]),
        "--min-trades-val",
        str(args.min_trades_val),
        "--max-flip-rate",
        str(args.max_flip_rate),
        "--walkforward-windows",
        str(args.walkforward_windows),
        "--live-shadow-summary-path",
        str(args.live_shadow_summary_path),
        "--setup-threshold-start",
        "0.20",
        "--setup-threshold-end",
        "0.85",
        "--setup-threshold-step",
        "0.03",
        "--direction-threshold-start",
        "0.55",
        "--direction-threshold-end",
        "0.82",
        "--direction-threshold-step",
        "0.02",
        "--setup-threshold-multiplier",
        str(cfg["setup_threshold_multiplier"]),
        "--max-bad-fade-rate",
        str(cfg["max_bad_fade_rate"]),
        "--entry-trend-filter",
        str(cfg.get("entry_trend_filter", "none")),
        "--min-signal-persistence-bars",
        str(cfg.get("min_signal_persistence_bars", 1)),
        "--cooldown-bars-after-flip",
        str(cfg.get("cooldown_bars_after_flip", 0)),
        "--commission-per-contract",
        "2.0",
        "--slippage-ticks",
        "1.0",
        "--n-estimators",
        str(args.n_estimators),
        "--close-giveback-close-r",
        str(cfg["close_giveback_close_r"]),
        "--close-stall-bars",
        str(cfg["close_stall_bars"]),
        "--close-severe-adverse-r",
        str(cfg["close_severe_adverse_r"]),
        "--generation",
        str(generation),
        "--parent-tag",
        parent_tag,
        "--promotion-target-tag",
        parent_tag,
        "--promotion-result",
        "pending",
    ]
    if args.require_safe_shadow_pass:
        cmd.append("--require-safe-shadow-pass")
    if cfg.get("stack_setup_prob"):
        cmd.append("--stack_setup_prob")
    if bool(cfg.get("htf_trend_aware")):
        cmd.append("--htf-trend-aware")
    mtf_timeframes = cfg.get("mtf_timeframes") or []
    if isinstance(mtf_timeframes, str):
        mtf_arg = mtf_timeframes
    else:
        mtf_arg = ",".join(str(item) for item in mtf_timeframes if str(item).strip())
    if mtf_arg:
        cmd.extend(["--mtf-timeframes", mtf_arg])
    return cmd


def _validate_cmd(artifact_root: Path, tag: str) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "tools" / "validate_features.py"),
        "--manifest",
        str(artifact_root / tag / "manifest.json"),
    ]


def _benchmark_cmd(args: argparse.Namespace, summary_dir: Path, generation: int, candidate_tags: List[str]) -> List[str]:
    tags: List[str] = []
    for tag in [str(args.deployment_baseline_tag or args.baseline_tag), str(args.baseline_tag), *candidate_tags]:
        if tag and tag not in tags:
            tags.append(tag)
    cmd = [
        sys.executable,
        "tools/benchmark_phase2_models.py",
        "--csv",
        str(args.csv),
        "--artifact-root",
        str(args.artifact_root),
        "--baseline-tag",
        str(args.deployment_baseline_tag or args.baseline_tag),
        "--legacy-baseline-tag",
        str(args.baseline_tag),
        "--test-start",
        str(args.test_start),
        "--test-end",
        str(args.test_end),
        "--min-trades-floor",
        str(args.min_trades_floor),
        "--out-dir",
        str(summary_dir / f"benchmark_generation_{generation:02d}"),
        "--tags",
    ]
    cmd.extend(tags)
    return cmd


def _load_resume_state(args: argparse.Namespace, summary_dir: Path, artifact_root: Path) -> Optional[Dict[str, Any]]:
    if args.fresh_run:
        return None
    registry_path = summary_dir / "generation_registry.json"
    registry = _read_json(registry_path, None)
    if not isinstance(registry, dict):
        return None
    current_tag = str(registry.get("current_champion_tag") or "").strip()
    if not current_tag:
        return None
    scoreboard_rows = _read_json(summary_dir / "generation_scoreboard.json", [])
    if not isinstance(scoreboard_rows, list):
        scoreboard_rows = []
    champion_history = _read_json(summary_dir / "champion_history.json", [])
    if not isinstance(champion_history, list):
        champion_history = []
    champion = _evaluate_tag(args, artifact_root, current_tag) if args.execute else _seed_champion(args)
    champion["generation"] = int(
        registry.get("current_generation")
        or registry.get("last_completed_generation")
        or _last_completed_generation(registry)
    )
    champion["parent_tag"] = None
    champion["promoted"] = True
    return {
        "registry": registry,
        "scoreboard_rows": scoreboard_rows,
        "champion_history": champion_history,
        "champion": champion,
        "last_completed_generation": int(champion.get("generation") or 0),
    }


def _persist_run_state(
    *,
    summary_dir: Path,
    registry: Dict[str, Any],
    champion_history: List[Dict[str, Any]],
    scoreboard_rows: List[Dict[str, Any]],
    champion: Dict[str, Any],
    artifact_root: Path,
    baseline_tag: str,
) -> None:
    registry["current_champion_tag"] = champion["tag"]
    registry["current_generation"] = int(champion.get("generation") or 0)
    registry["last_completed_generation"] = _last_completed_generation(registry)
    _write_json(summary_dir / "generation_registry.json", registry)
    _write_json(summary_dir / "champion_history.json", champion_history)
    _write_scoreboard(summary_dir / "generation_scoreboard.json", scoreboard_rows)
    _write_champion_alias(
        artifact_root / "current_champion.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "tag": champion["tag"],
            "generation": champion.get("generation"),
            "baseline_tag": baseline_tag,
            "deployment_baseline_tag": registry.get("deployment_baseline_tag"),
            "search_bias": champion.get("search_bias"),
        },
    )


def _write_benchmark_backrefs(artifact_root: Path, benchmark_dir: Path) -> None:
    payload = _read_json(benchmark_dir / "phase2_benchmark.json", None)
    if not isinstance(payload, dict):
        return
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        tag = str(row.get("tag") or "").strip()
        if not tag:
            continue
        manifest_path = _candidate_manifest_path(artifact_root, tag)
        if not manifest_path.exists():
            continue
        updates = {
            "benchmark_score": row.get("score"),
            "benchmark_rank": row.get("rank"),
            "benchmark_tier": row.get("tier"),
            "benchmark_failure_family": row.get("failure_family"),
            "benchmark_deployable": row.get("deployable"),
            "benchmark_slippage_pass": row.get("slippage_pass"),
            "benchmark_generated_at": payload.get("generated_at"),
        }
        _update_candidate_manifest(manifest_path, updates)


def _apply_benchmark_rows_to_scoreboard(scoreboard_rows: List[Dict[str, Any]], benchmark_dir: Path) -> None:
    payload = _read_json(benchmark_dir / "phase2_benchmark.json", None)
    if not isinstance(payload, dict):
        return
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        return
    by_tag = {str(row.get("tag") or ""): row for row in rows if isinstance(row, dict)}
    for item in scoreboard_rows:
        tag = str(item.get("tag") or "")
        if not tag or tag not in by_tag:
            continue
        bench = by_tag[tag]
        item["benchmark_score"] = bench.get("score")
        item["benchmark_rank"] = bench.get("rank")
        item["benchmark_tier"] = bench.get("tier")
        item["benchmark_failure_family"] = bench.get("failure_family")
        item["benchmark_deployable"] = bench.get("deployable")
        item["benchmark_slippage_pass"] = bench.get("slippage_pass")


def _apply_benchmark_rows(rows: List[Dict[str, Any]], benchmark_dir: Path) -> None:
    payload = _read_json(benchmark_dir / "phase2_benchmark.json", None)
    if not isinstance(payload, dict):
        return
    bench_rows = payload.get("rows") or []
    if not isinstance(bench_rows, list):
        return
    by_tag = {str(row.get("tag") or ""): row for row in bench_rows if isinstance(row, dict)}
    for item in rows:
        tag = str(item.get("tag") or "")
        bench = by_tag.get(tag)
        if not tag or not bench:
            continue
        item["benchmark_score"] = bench.get("score")
        item["benchmark_rank"] = bench.get("rank")
        item["benchmark_tier"] = bench.get("tier")
        item["benchmark_failure_family"] = bench.get("failure_family")
        item["benchmark_deployable"] = bench.get("deployable")
        item["benchmark_slippage_pass"] = bench.get("slippage_pass")


def _last_completed_generation(registry: Dict[str, Any]) -> int:
    return max(
        (int(item.get("generation") or 0) for item in registry.get("generations", []) if isinstance(item, dict)),
        default=0,
    )


def _load_generation_feedback(summary_dir: Path, generation: int) -> Dict[str, Any]:
    if generation <= 1:
        return {
            "source_generation": None,
            "failure_counts": {},
            "recommended_next_biases": [],
            "top_rejected": None,
            "top_rejected_threshold_diagnostics": {},
        }
    benchmark_path = summary_dir / f"benchmark_generation_{generation - 1:02d}" / "phase2_benchmark.json"
    payload = _read_json(benchmark_path, None)
    if not isinstance(payload, dict):
        return {
            "source_generation": None,
            "failure_counts": {},
            "recommended_next_biases": [],
            "top_rejected": None,
            "top_rejected_threshold_diagnostics": {},
        }
    guidance = payload.get("guidance") or {}
    rows = payload.get("rows") or []
    rejected_rows = [
        row for row in rows
        if isinstance(row, dict) and str(row.get("tier") or "") == "C"
    ]
    top_rejected = rejected_rows[0] if rejected_rows else None
    return {
        "source_generation": generation - 1,
        "failure_counts": dict(guidance.get("failure_counts") or {}) if isinstance(guidance, dict) else {},
        "recommended_next_biases": list(guidance.get("recommended_next_biases") or []) if isinstance(guidance, dict) else [],
        "top_rejected": top_rejected,
        "top_rejected_threshold_diagnostics": dict((top_rejected or {}).get("threshold_diagnostics") or {}),
    }


def _dominant_failure_family(failure_counts: Dict[str, Any]) -> Optional[str]:
    normalized: Dict[str, int] = {}
    for key, value in failure_counts.items():
        try:
            normalized[str(key)] = int(value or 0)
        except Exception:
            normalized[str(key)] = 0
    candidates = ["low_trades", "threshold_rejected", "bad_fade", "high_flip"]
    winner = None
    best = 0
    for key in candidates:
        count = normalized.get(key, 0)
        if count > best:
            best = count
            winner = key
    return winner if best > 0 else None


def _resolve_search_bias(args: argparse.Namespace, feedback: Dict[str, Any]) -> str:
    if str(args.search_bias) != "auto":
        return str(args.search_bias)
    dominant = _dominant_failure_family(dict(feedback.get("failure_counts") or {}))
    if dominant in {"low_trades", "threshold_rejected"}:
        return "trade_recovery"
    if dominant == "bad_fade":
        return "fade_control"
    return "balanced"


def _candidate_manifest_metadata(cfg: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "search_bias": cfg.get("search_bias"),
        "source_failure_family": cfg.get("source_failure_family"),
        "source_generation": cfg.get("source_generation"),
    }
    if metadata["source_failure_family"] is None:
        metadata["source_failure_family"] = _dominant_failure_family(dict(feedback.get("failure_counts") or {}))
    return metadata


def _candidate_slot_incomplete(artifact_root: Path, tag: str) -> bool:
    candidate_dir = artifact_root / tag
    if not candidate_dir.exists():
        return False
    manifest_path = candidate_dir / "manifest.json"
    if not manifest_path.exists():
        return True
    payload = _read_json(manifest_path, None)
    if not isinstance(payload, dict):
        return True
    return not payload.get("config") and not payload.get("thresholds") and not payload.get("metrics")


def _candidate_space(args: argparse.Namespace, generation: int, champion: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = champion.get("config") or {}
    selectivity = champion.get("setup_selectivity_stats") or {}
    behavior = champion.get("behavior_audit_test") or {}
    slip1 = (champion.get("slippage") or {}).get("slip_1") or {}
    feedback = _load_generation_feedback(Path(args.summary_dir).expanduser().resolve(), generation)
    search_bias = _resolve_search_bias(args, feedback)

    base_label_mode = str(cfg.get("label_mode") or "horizon")
    close_replay = dict(cfg.get("close_replay") or {})
    close_giveback_close_r = float(close_replay.get("pnl_giveback_close_r") or 0.5)
    close_stall_bars = int(close_replay.get("pnl_stall_bars") or 4)
    close_severe_adverse_r = float(close_replay.get("pnl_severe_adverse_r") or 0.9)

    trade_rate = float(selectivity.get("trade_rate") or 0.0)
    bad_fade_rate = float(behavior.get("countertrend_rate") or 0.0)
    trades = int(slip1.get("trade_count") or 0)
    max_loss_streak = int(champion.get("max_consecutive_losses") or 0)

    if max_loss_streak >= 4:
        close_giveback_close_r = max(0.25, close_giveback_close_r - 0.10)
        close_stall_bars = max(2, close_stall_bars - 1)
        close_severe_adverse_r = max(0.55, close_severe_adverse_r - 0.10)

    if search_bias == "fade_control" or bad_fade_rate > args.max_bad_fade_rate:
        horizons = [8, 10, 12]
        label_thresholds = [0.0012, 0.0015]
        base_label_mode = "exec"
    else:
        horizons = [6, 8]
        label_thresholds = [0.0010, 0.0012, 0.0014]
    objectives = ["sharpe", "calmar"]
    valid_combo_count = int((feedback.get("top_rejected_threshold_diagnostics") or {}).get("valid_combos") or 0)

    if generation == 1 or search_bias == "trade_recovery":
        setup_mults = [1.10, 1.20, 1.30, 1.40]
    elif search_bias == "fade_control":
        setup_mults = [1.10, 1.20, 1.20, 1.30]
    else:
        setup_mults = [1.10, 1.20, 1.30, 1.40]

    if valid_combo_count < 10:
        setup_mults = [value for value in setup_mults if value <= 1.50]
    else:
        setup_mults = sorted({*setup_mults, 1.50})

    if search_bias == "fade_control" or bad_fade_rate > args.max_bad_fade_rate:
        close_profiles = [
            (round(max(0.25, close_giveback_close_r - 0.10), 2), max(2, close_stall_bars - 1), round(max(0.55, close_severe_adverse_r - 0.10), 2)),
            (round(max(0.25, close_giveback_close_r - 0.05), 2), close_stall_bars, round(max(0.55, close_severe_adverse_r - 0.05), 2)),
        ]
    elif trade_rate < 0.10 or trades < args.min_trades_floor:
        close_profiles = [
            (round(close_giveback_close_r, 2), close_stall_bars, round(close_severe_adverse_r, 2)),
            (round(close_giveback_close_r + 0.05, 2), close_stall_bars + 1, round(min(1.0, close_severe_adverse_r + 0.05), 2)),
        ]
    else:
        close_profiles = [
            (round(close_giveback_close_r, 2), close_stall_bars, round(close_severe_adverse_r, 2)),
            (round(max(0.25, close_giveback_close_r - 0.05), 2), max(2, close_stall_bars - 1), round(max(0.55, close_severe_adverse_r - 0.05), 2)),
        ]

    source_failure_family = _dominant_failure_family(dict(feedback.get("failure_counts") or {}))
    recommended_next_biases = [str(item) for item in (feedback.get("recommended_next_biases") or [])]

    if search_bias == "fade_control" or bad_fade_rate > args.max_bad_fade_rate:
        sequence = [
            {"horizon": 8, "label_threshold": 0.0015, "label_max_hold_bars": 18, "setup_threshold_multiplier": 1.10, "threshold_objective": "sharpe", "stack_setup_prob": True, "htf_trend_aware": True},
            {"horizon": 8, "label_threshold": 0.0015, "label_max_hold_bars": 24, "setup_threshold_multiplier": 1.20, "threshold_objective": "sharpe", "stack_setup_prob": True, "htf_trend_aware": True},
            {"horizon": 10, "label_threshold": 0.0015, "label_max_hold_bars": 18, "setup_threshold_multiplier": 1.20, "threshold_objective": "calmar", "stack_setup_prob": True, "htf_trend_aware": True},
            {"horizon": 10, "label_threshold": 0.0012, "label_max_hold_bars": 24, "setup_threshold_multiplier": 1.30, "threshold_objective": "sharpe", "stack_setup_prob": True, "htf_trend_aware": True},
            {"horizon": 12, "label_threshold": 0.0015, "label_max_hold_bars": 24, "setup_threshold_multiplier": 1.20, "threshold_objective": "calmar", "stack_setup_prob": True, "htf_trend_aware": True},
            {"horizon": 12, "label_threshold": 0.0012, "label_max_hold_bars": 18, "setup_threshold_multiplier": 1.10, "threshold_objective": "sharpe", "stack_setup_prob": True, "htf_trend_aware": True},
        ]
    else:
        sequence = [
            {"horizon": 6, "label_threshold": 0.0010, "setup_threshold_multiplier": 1.10, "threshold_objective": "sharpe", "stack_setup_prob": False},
            {"horizon": 6, "label_threshold": 0.0012, "setup_threshold_multiplier": 1.20, "threshold_objective": "sharpe", "stack_setup_prob": False},
            {"horizon": 8, "label_threshold": 0.0010, "setup_threshold_multiplier": 1.20, "threshold_objective": "calmar", "stack_setup_prob": False},
            {"horizon": 6, "label_threshold": 0.0014, "setup_threshold_multiplier": 1.30, "threshold_objective": "sharpe", "stack_setup_prob": False},
            {"horizon": 8, "label_threshold": 0.0012, "setup_threshold_multiplier": 1.30, "threshold_objective": "calmar", "stack_setup_prob": False},
            {"horizon": 6, "label_threshold": 0.0010, "setup_threshold_multiplier": 1.40, "threshold_objective": "sharpe", "stack_setup_prob": False},
            {"horizon": 8, "label_threshold": 0.0014, "setup_threshold_multiplier": 1.40, "threshold_objective": "calmar", "stack_setup_prob": False},
            {"horizon": 6, "label_threshold": 0.0012, "setup_threshold_multiplier": 1.20, "threshold_objective": "sharpe", "stack_setup_prob": True},
        ]

    variants: List[Dict[str, Any]] = []
    for idx, item in enumerate(sequence):
        if item["horizon"] not in horizons or item["label_threshold"] not in label_thresholds or item["threshold_objective"] not in objectives:
            continue
        if item["setup_threshold_multiplier"] not in setup_mults:
            continue
        close_giveback, close_stall, close_severe = close_profiles[idx % len(close_profiles)]
        variants.append(
            {
                "label_mode": base_label_mode,
                "horizon": item["horizon"],
                "label_threshold": item["label_threshold"],
                "label_max_hold_bars": int(item.get("label_max_hold_bars") or item["horizon"]),
                "threshold_objective": item["threshold_objective"],
                "stack_setup_prob": bool(item["stack_setup_prob"]),
                "htf_trend_aware": bool(item.get("htf_trend_aware", False)),
                "setup_threshold_multiplier": round(float(item["setup_threshold_multiplier"]), 2),
                "max_bad_fade_rate": round(args.max_bad_fade_rate, 3),
                "entry_trend_filter": "vwap_ema" if (search_bias == "fade_control" or bad_fade_rate > args.max_bad_fade_rate) else "none",
                "min_signal_persistence_bars": 2 if (search_bias == "fade_control" or bad_fade_rate > args.max_bad_fade_rate) else 1,
                "cooldown_bars_after_flip": 3 if (search_bias == "fade_control" or bad_fade_rate > args.max_bad_fade_rate) else 0,
                "close_giveback_close_r": close_giveback,
                "close_stall_bars": close_stall,
                "close_severe_adverse_r": close_severe,
                "search_bias": search_bias,
                "source_failure_family": source_failure_family,
                "source_generation": feedback.get("source_generation"),
                "recommended_next_biases": recommended_next_biases,
            }
        )

    return variants[: max(1, int(args.generation_budget))]


def _candidate_passes_gates(args: argparse.Namespace, champion: Dict[str, Any], candidate: Dict[str, Any]) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    champ_slip1 = (champion.get("slippage") or {}).get("slip_1") or {}
    cand_slip1 = (candidate.get("slippage") or {}).get("slip_1") or {}
    cand_bad_fade = float(((candidate.get("behavior_audit_test") or {}).get("countertrend_rate") or 0.0))
    champ_bad_fade = float(((champion.get("behavior_audit_test") or {}).get("countertrend_rate") or 0.0))

    if candidate.get("rejected"):
        reasons.append(f"trainer rejected candidate: {candidate.get('rejected_reason')}")
    if candidate.get("promotion_blocked"):
        reasons.append(str(candidate.get("promotion_blocked_reason") or "promotion blocked by trainer gate"))
    if candidate.get("benchmark_deployable") is False:
        reasons.append("benchmark marked candidate non-deployable")
    if candidate.get("benchmark_slippage_pass") is False:
        reasons.append("benchmark slippage pass failed")
    if args.require_safe_shadow_pass and not bool(candidate.get("live_shadow_pass")):
        reasons.append("live shadow gate failed")
    if args.require_slippage_pass and not bool(candidate.get("slippage_pass")):
        reasons.append("failed slippage robustness at 0/1/2 ticks")
    if cand_bad_fade > float(args.max_bad_fade_rate):
        reasons.append(f"bad fade rate {cand_bad_fade:.3f} exceeds {args.max_bad_fade_rate:.3f}")
    if cand_bad_fade > champ_bad_fade and champ_bad_fade > 0.0:
        reasons.append(f"bad fade rate worsened vs champion ({cand_bad_fade:.3f} > {champ_bad_fade:.3f})")
    if int(cand_slip1.get("trade_count") or 0) < int(args.min_trades_floor):
        reasons.append("OOS trade count below deployment floor")
    if float(cand_slip1.get("total_pnl_usd") or 0.0) <= 0.0:
        reasons.append("OOS pnl is not positive")
    if float(cand_slip1.get("profit_factor") or 0.0) < 1.0:
        reasons.append("OOS profit factor below 1.0")
    if not math.isfinite(float(cand_slip1.get("sharpe") or 0.0)):
        reasons.append("OOS sharpe is not finite")
    if abs(float(cand_slip1.get("max_drawdown") or 0.0)) > abs(float(champ_slip1.get("max_drawdown") or 0.0)) * float(args.drawdown_tolerance):
        reasons.append("drawdown beyond tolerance vs champion")
    if int(candidate.get("max_consecutive_losses") or 0) > int(champion.get("max_consecutive_losses") or 0) + int(args.loss_streak_tolerance):
        reasons.append("loss streak worse than champion tolerance")
    return len(reasons) == 0, reasons


def _promote_best(args: argparse.Namespace, champion: Dict[str, Any], candidates: List[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    ranked: List[Dict[str, Any]] = []
    for candidate in candidates:
        ok, reasons = _candidate_passes_gates(args, champion, candidate)
        candidate["promotion_pass"] = ok
        candidate["promotion_reasons"] = reasons
        ranked.append(candidate)
    passed = [row for row in ranked if row.get("promotion_pass")]
    if not passed:
        return None, ranked
    passed.sort(
        key=lambda row: (
            float(row.get("benchmark_score") if row.get("benchmark_score") is not None else row.get("risk_score") or 0.0),
            -int(row.get("benchmark_rank") or 999999),
            float(((row.get("slippage") or {}).get("slip_1") or {}).get("profit_factor") or 0.0),
        ),
        reverse=True,
    )
    return passed[0], ranked


def _write_scoreboard(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    standalone_rows: List[Dict[str, Any]] = []
    for row in rows:
        slip1 = (row.get("slippage") or {}).get("slip_1") or {}
        behavior = row.get("behavior_audit_test") or {}
        selectivity = row.get("setup_selectivity_stats") or {}
        standalone_rows.append(
            {
                "tag": row.get("tag"),
                "generation": row.get("generation"),
                "parent_tag": row.get("parent_tag"),
                "candidate_status": row.get("candidate_status"),
                "promoted": row.get("promoted"),
                "promotion_pass": row.get("promotion_pass"),
                "risk_score": row.get("risk_score"),
                "profit_factor_slip1": slip1.get("profit_factor"),
                "sharpe_slip1": slip1.get("sharpe"),
                "pnl_slip1": slip1.get("total_pnl_usd"),
                "max_dd_slip1": slip1.get("max_drawdown"),
                "trade_count_slip1": slip1.get("trade_count"),
                "max_consecutive_losses": row.get("max_consecutive_losses"),
                "bad_fade_rate": behavior.get("countertrend_rate"),
                "setup_trade_rate": selectivity.get("trade_rate"),
                "slippage_pass": row.get("slippage_pass"),
                "promotion_reasons": "; ".join(str(x) for x in (row.get("promotion_reasons") or [])),
                "rejected_reason": row.get("rejected_reason"),
                "benchmark_score": row.get("benchmark_score"),
                "benchmark_rank": row.get("benchmark_rank"),
                "threshold_valid_combos": (row.get("threshold_diagnostics") or {}).get("valid_combos"),
                "threshold_rejected_low_trades": (row.get("threshold_diagnostics") or {}).get("rejected_low_trades"),
                "threshold_rejected_high_flip": (row.get("threshold_diagnostics") or {}).get("rejected_high_flip"),
                "threshold_rejected_bad_fade": (row.get("threshold_diagnostics") or {}).get("rejected_bad_fade"),
            }
        )
    _write_json(path, standalone_rows)
    csv_path = path.with_suffix(".csv")
    if standalone_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(standalone_rows[0].keys()))
            writer.writeheader()
            writer.writerows(standalone_rows)


def _write_generation_report(path: Path, generation: int, baseline_tag: str, champion_before: str, winner: Optional[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(f"# Generation {generation:02d} Report")
    lines.append("")
    lines.append(f"- Baseline: `{baseline_tag}`")
    lines.append(f"- Champion entering generation: `{champion_before}`")
    lines.append(f"- Winner: `{winner.get('tag') if winner else champion_before}`")
    lines.append(f"- Promoted: `{'yes' if winner else 'no'}`")
    lines.append("")
    lines.append("| tag | pass | promoted | score | pf@1 | sharpe@1 | pnl@1 | dd@1 | max loss streak | bad fade | setup trade rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(candidates, key=lambda item: float(item.get("risk_score") or 0.0), reverse=True):
        slip1 = (row.get("slippage") or {}).get("slip_1") or {}
        behavior = row.get("behavior_audit_test") or {}
        selectivity = row.get("setup_selectivity_stats") or {}
        lines.append(
            "| {tag} | {pass_flag} | {promoted} | {score:.2f} | {pf:.2f} | {sharpe:.2f} | {pnl:.2f} | {dd:.2f} | {losses} | {fade:.3f} | {trade_rate:.3f} |".format(
                tag=row.get("tag"),
                pass_flag="yes" if row.get("promotion_pass") else "no",
                promoted="yes" if row.get("promoted") else "no",
                score=float(row.get("risk_score") or 0.0),
                pf=float(slip1.get("profit_factor") or 0.0),
                sharpe=float(slip1.get("sharpe") or 0.0),
                pnl=float(slip1.get("total_pnl_usd") or 0.0),
                dd=float(slip1.get("max_drawdown") or 0.0),
                losses=int(row.get("max_consecutive_losses") or 0),
                fade=float(behavior.get("countertrend_rate") or 0.0),
                trade_rate=float(selectivity.get("trade_rate") or 0.0),
            )
        )
        reasons = row.get("promotion_reasons") or []
        if reasons:
            lines.append(f"\n  Promotion notes for `{row.get('tag')}`: {', '.join(str(x) for x in reasons)}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_champion_alias(path: Path, payload: Dict[str, Any]) -> None:
    _write_json(path, payload)


def main() -> int:
    args = _parse_args()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    summary_dir = Path(args.summary_dir).expanduser().resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    seed_tag = str(args.deployment_baseline_tag or args.baseline_tag)
    resume_state = _load_resume_state(args, summary_dir, artifact_root)
    if resume_state:
        champion = resume_state["champion"]
        registry = resume_state["registry"]
        champion_history = resume_state["champion_history"]
        scoreboard_rows = resume_state["scoreboard_rows"] or [champion]
        start_generation = int(resume_state["last_completed_generation"]) + 1
    else:
        champion = _evaluate_tag(args, artifact_root, seed_tag) if args.execute else _seed_champion(args)
        champion["generation"] = 0
        champion["parent_tag"] = None
        champion["promoted"] = True
        registry = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_tag": args.baseline_tag,
            "deployment_baseline_tag": seed_tag,
            "search_bias": args.search_bias,
            "promote_on": args.promote_on,
            "drawdown_tolerance": args.drawdown_tolerance,
            "loss_streak_tolerance": args.loss_streak_tolerance,
            "max_bad_fade_rate": args.max_bad_fade_rate,
            "max_flip_rate": args.max_flip_rate,
            "min_trades_val": args.min_trades_val,
            "require_slippage_pass": args.require_slippage_pass,
            "generations": [],
            "current_champion_tag": seed_tag,
            "current_generation": 0,
            "last_completed_generation": 0,
        }
        champion_history = [
            {
                "generation": 0,
                "tag": seed_tag,
                "source": "deployment_baseline",
                "promoted": True,
                "risk_score": champion.get("risk_score"),
                "legacy_baseline_tag": args.baseline_tag,
            }
        ]
        scoreboard_rows = [champion]
        _persist_run_state(
            summary_dir=summary_dir,
            registry=registry,
            champion_history=champion_history,
            scoreboard_rows=scoreboard_rows,
            champion=champion,
            artifact_root=artifact_root,
            baseline_tag=args.baseline_tag,
        )
        start_generation = 1

    end_generation = start_generation + max(0, int(args.generations)) - 1
    for generation in range(start_generation, end_generation + 1):
        champion_before = champion["tag"]
        candidate_cfgs = _candidate_space(args, generation, champion)
        candidate_rows: List[Dict[str, Any]] = []
        generation_meta: Dict[str, Any] = {
            "generation": generation,
            "parent_tag": champion_before,
            "candidate_tags": [],
            "winner_tag": None,
            "promoted": False,
            "promotion_reasons": [],
        }

        for idx, cfg in enumerate(candidate_cfgs, start=1):
            tag = f"{args.tag_prefix}_gen{generation:02d}_cand{idx:02d}"
            generation_meta["candidate_tags"].append(tag)
            metadata_updates = _candidate_manifest_metadata(cfg, _load_generation_feedback(summary_dir, generation))
            if _candidate_slot_incomplete(artifact_root, tag):
                reason = "failed_incomplete: stale_candidate_slot_without_finalized_manifest"
                _record_candidate_failure(
                    artifact_root,
                    tag,
                    generation=generation,
                    parent_tag=champion_before,
                    reason=reason,
                    status="failed_incomplete",
                )
                row = _failed_candidate_row(
                    tag=tag,
                    artifact_root=artifact_root,
                    generation=generation,
                    parent_tag=champion_before,
                    cfg=cfg,
                    reason=reason,
                    status="failed_incomplete",
                )
                row["search_bias"] = cfg.get("search_bias")
                row["source_failure_family"] = cfg.get("source_failure_family")
                row["source_generation"] = cfg.get("source_generation")
                scoreboard_rows.append(row)
                candidate_rows.append(row)
                continue
            _set_candidate_status(
                artifact_root,
                tag,
                generation=generation,
                parent_tag=champion_before,
                promotion_target_tag=champion_before,
                status="training",
                promotion_result="pending",
                extra_updates={"rejected": False, "rejected_reason": None, **metadata_updates},
            )
            ok, _, err = _safe_run(_train_cmd(args, tag, cfg, generation, champion_before), execute=args.execute)
            if not ok:
                reason = f"training_failed: {err}"
                _record_candidate_failure(
                    artifact_root,
                    tag,
                    generation=generation,
                    parent_tag=champion_before,
                    reason=reason,
                    status="failed",
                    thresholds={},
                    threshold_diagnostics={},
                )
                row = _failed_candidate_row(
                    tag=tag,
                    artifact_root=artifact_root,
                    generation=generation,
                    parent_tag=champion_before,
                    cfg=cfg,
                    reason=reason,
                    status="failed",
                )
                row["search_bias"] = cfg.get("search_bias")
                row["source_failure_family"] = cfg.get("source_failure_family")
                row["source_generation"] = cfg.get("source_generation")
                scoreboard_rows.append(row)
                candidate_rows.append(row)
                continue
            if not args.execute:
                continue
            manifest_path = _candidate_manifest_path(artifact_root, tag)
            if not manifest_path.exists():
                reason = "failed_incomplete: manifest_missing_after_training"
                _record_candidate_failure(
                    artifact_root,
                    tag,
                    generation=generation,
                    parent_tag=champion_before,
                    reason=reason,
                    status="failed_incomplete",
                )
                row = _failed_candidate_row(
                    tag=tag,
                    artifact_root=artifact_root,
                    generation=generation,
                    parent_tag=champion_before,
                    cfg=cfg,
                    reason=reason,
                    status="failed_incomplete",
                )
                row["search_bias"] = cfg.get("search_bias")
                row["source_failure_family"] = cfg.get("source_failure_family")
                row["source_generation"] = cfg.get("source_generation")
                scoreboard_rows.append(row)
                candidate_rows.append(row)
                continue
            manifest_payload = _read_json(manifest_path, {})
            placeholder_manifest = (
                isinstance(manifest_payload, dict)
                and str(manifest_payload.get("candidate_status") or "") == "training"
                and not manifest_payload.get("config")
                and not manifest_payload.get("thresholds")
                and not manifest_payload.get("metrics")
            )
            if placeholder_manifest:
                reason = "failed_incomplete: trainer_did_not_finalize_manifest"
                _record_candidate_failure(
                    artifact_root,
                    tag,
                    generation=generation,
                    parent_tag=champion_before,
                    reason=reason,
                    status="failed_incomplete",
                )
                row = _failed_candidate_row(
                    tag=tag,
                    artifact_root=artifact_root,
                    generation=generation,
                    parent_tag=champion_before,
                    cfg=cfg,
                    reason=reason,
                    status="failed_incomplete",
                )
                row["search_bias"] = cfg.get("search_bias")
                row["source_failure_family"] = cfg.get("source_failure_family")
                row["source_generation"] = cfg.get("source_generation")
                scoreboard_rows.append(row)
                candidate_rows.append(row)
                continue
            _set_candidate_status(
                artifact_root,
                tag,
                generation=generation,
                parent_tag=champion_before,
                promotion_target_tag=champion_before,
                status="trained",
                promotion_result="pending",
                extra_updates=metadata_updates,
            )
            ok, _, err = _safe_run(_validate_cmd(artifact_root, tag), execute=True)
            if not ok:
                reason = f"validation_failed: {err}"
                manifest_payload = _read_json(manifest_path, {})
                _record_candidate_failure(
                    artifact_root,
                    tag,
                    generation=generation,
                    parent_tag=champion_before,
                    reason=reason,
                    status="failed",
                    thresholds=(manifest_payload.get("thresholds") if isinstance(manifest_payload, dict) else {}) or {},
                    threshold_diagnostics=(manifest_payload.get("threshold_diagnostics") if isinstance(manifest_payload, dict) else {}) or {},
                )
                row = _failed_candidate_row(
                    tag=tag,
                    artifact_root=artifact_root,
                    generation=generation,
                    parent_tag=champion_before,
                    cfg=cfg,
                    reason=reason,
                    status="failed",
                    thresholds=(manifest_payload.get("thresholds") if isinstance(manifest_payload, dict) else {}) or {},
                    threshold_diagnostics=(manifest_payload.get("threshold_diagnostics") if isinstance(manifest_payload, dict) else {}) or {},
                )
                row["search_bias"] = cfg.get("search_bias")
                row["source_failure_family"] = cfg.get("source_failure_family")
                row["source_generation"] = cfg.get("source_generation")
                scoreboard_rows.append(row)
                candidate_rows.append(row)
                continue
            _set_candidate_status(
                artifact_root,
                tag,
                generation=generation,
                parent_tag=champion_before,
                promotion_target_tag=champion_before,
                status="validated",
                promotion_result="pending",
                extra_updates=metadata_updates,
            )
            try:
                manifest = _load_manifest(tag, artifact_root)
            except Exception as exc:
                reason = f"failed_incomplete: manifest_load_failed: {exc}"
                _record_candidate_failure(
                    artifact_root,
                    tag,
                    generation=generation,
                    parent_tag=champion_before,
                    reason=reason,
                    status="failed_incomplete",
                )
                row = _failed_candidate_row(
                    tag=tag,
                    artifact_root=artifact_root,
                    generation=generation,
                    parent_tag=champion_before,
                    cfg=cfg,
                    reason=reason,
                    status="failed_incomplete",
                )
                row["search_bias"] = cfg.get("search_bias")
                row["source_failure_family"] = cfg.get("source_failure_family")
                row["source_generation"] = cfg.get("source_generation")
                scoreboard_rows.append(row)
                candidate_rows.append(row)
                continue
            if bool(manifest.get("rejected")):
                manifest_path = Path(str(manifest["_manifest_path"]))
                _update_candidate_manifest(
                    manifest_path,
                    {
                        "generation": generation,
                        "parent_tag": champion_before,
                        "promotion_target_tag": champion_before,
                        "promotion_result": "evaluated_skipped_rejected",
                        "candidate_status": "rejected",
                        "status_updated_at": datetime.now(timezone.utc).isoformat(),
                        **metadata_updates,
                    },
                )
                row = _rejected_candidate_row(manifest, generation=generation, parent_tag=champion_before, cfg=cfg)
                row["search_bias"] = cfg.get("search_bias")
                row["source_failure_family"] = cfg.get("source_failure_family")
                row["source_generation"] = cfg.get("source_generation")
                scoreboard_rows.append(row)
                candidate_rows.append(row)
                continue
            row = _evaluate_tag(args, artifact_root, tag)
            row["generation"] = generation
            row["parent_tag"] = champion_before
            row["candidate_config"] = cfg
            row["search_bias"] = cfg.get("search_bias")
            row["source_failure_family"] = cfg.get("source_failure_family")
            row["source_generation"] = cfg.get("source_generation")
            manifest_path = Path(str(row["manifest_path"]))
            _update_candidate_manifest(
                manifest_path,
                {
                    "generation": generation,
                    "parent_tag": champion_before,
                    "promotion_target_tag": champion_before,
                    "promotion_result": "evaluated",
                    "candidate_status": "evaluated",
                    "status_updated_at": datetime.now(timezone.utc).isoformat(),
                    **metadata_updates,
                },
            )
            candidate_rows.append(row)
            scoreboard_rows.append(row)

        if not args.execute:
            if args.benchmark_after_generation:
                _run(_benchmark_cmd(args, summary_dir, generation, generation_meta["candidate_tags"]), execute=False)
            continue

        benchmark_dir = summary_dir / f"benchmark_generation_{generation:02d}"
        _run(_benchmark_cmd(args, summary_dir, generation, generation_meta["candidate_tags"]), execute=True)
        _write_benchmark_backrefs(artifact_root, benchmark_dir)
        _apply_benchmark_rows(candidate_rows, benchmark_dir)
        _apply_benchmark_rows_to_scoreboard(scoreboard_rows, benchmark_dir)

        winner, ranked = _promote_best(args, champion, candidate_rows)
        if winner is not None:
            winner["promoted"] = True
            winner["candidate_status"] = "promoted"
            champion = winner
            champion["generation"] = generation
            generation_meta["winner_tag"] = winner["tag"]
            generation_meta["promoted"] = True
            generation_meta["promotion_reasons"] = ["passed risk-adjusted OOS gates", "selected by benchmark-backed ranking"]
            _update_candidate_manifest(
                Path(str(winner["manifest_path"])),
                {
                    "promotion_result": "promoted",
                    "promotion_target_tag": champion_before,
                    "candidate_status": "promoted",
                    "status_updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            champion_history.append(
                {
                    "generation": generation,
                    "tag": winner["tag"],
                    "source": "generation_winner",
                    "promoted": True,
                    "replaced": champion_before,
                    "risk_score": winner.get("risk_score"),
                    "benchmark_score": winner.get("benchmark_score"),
                }
            )
        else:
            generation_meta["winner_tag"] = champion_before
            generation_meta["promotion_reasons"] = ["no candidate beat current champion on hard gates after benchmark"]
            champion_history.append(
                {
                    "generation": generation,
                    "tag": champion_before,
                    "source": "incumbent_retained",
                    "promoted": False,
                    "risk_score": champion.get("risk_score"),
                }
            )

        for row in ranked:
            row.setdefault("promoted", False)
            if not row.get("promoted"):
                manifest_path = row.get("manifest_path")
                if manifest_path and Path(str(manifest_path)).exists():
                    _update_candidate_manifest(
                        Path(str(manifest_path)),
                        {
                            "promotion_result": "not_promoted",
                            "promotion_target_tag": champion_before,
                            "candidate_status": "not_promoted" if not row.get("rejected") else row.get("candidate_status") or "rejected",
                            "status_updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )

        registry["generations"].append(generation_meta)
        _write_generation_report(summary_dir / f"generation_{generation:02d}_report.md", generation, args.baseline_tag, champion_before, winner, ranked)
        registry["current_generation"] = generation
        registry["last_completed_generation"] = generation
        _persist_run_state(
            summary_dir=summary_dir,
            registry=registry,
            champion_history=champion_history,
            scoreboard_rows=scoreboard_rows,
            champion=champion,
            artifact_root=artifact_root,
            baseline_tag=args.baseline_tag,
        )
        _persist_run_state(
            summary_dir=summary_dir,
            registry=registry,
            champion_history=champion_history,
            scoreboard_rows=scoreboard_rows,
            champion=champion,
            artifact_root=artifact_root,
            baseline_tag=args.baseline_tag,
        )

    if args.execute:
        final_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_tag": args.baseline_tag,
            "deployment_baseline_tag": seed_tag,
            "final_champion_tag": champion["tag"],
            "current_champion": champion,
        }
        _write_json(summary_dir / "final_champion.json", final_payload)

    print("\nGenerational workflow notes:")
    print("1. Dry-run prints the training/validation commands for each generation.")
    print("2. Execute mode writes generation_registry.json, champion_history.json, generation_scoreboard.json, and per-generation markdown reports.")
    print("3. current_champion.json under artifacts/phase2/candidates points to the latest promoted lineage winner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


