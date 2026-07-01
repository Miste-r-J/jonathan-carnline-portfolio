from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

import pandas as pd

from na.bot import autonomy_test, journal_coach, market_dashboard, market_eval, preset_evolve


@dataclass
class TrainAllConfig:
    symbols: List[str]
    model_path: str
    preset_dir: str
    data_dir: str
    eval_results_dir: str
    evolution_results_dir: str
    autonomy_results_dir: str
    dashboard_out: str
    eval_config_name: str
    use_online_updates: bool
    model_save_dir: str | None
    num_evolution_windows: int
    evolution_window_days: int
    num_autonomy_windows: int
    autonomy_window_days: int
    baseline_dir: str
    store_baseline: bool
    preset_iterations: int
    quick_mode: bool


def default_config() -> TrainAllConfig:
    return TrainAllConfig(
        symbols=["ES", "6E", "GC", "NQ"],  # extend with other Topstep-compatible symbols as supported
        model_path="/home/ubuntu/Desktop/artifacts/really_good_model.joblib",
        preset_dir="/home/ubuntu/Desktop/preset_search_results",
        data_dir="/home/ubuntu/Desktop/market_eval_data",
        eval_results_dir="/home/ubuntu/Desktop/market_eval_results",
        evolution_results_dir="/home/ubuntu/Desktop/preset_evolution_results",
        autonomy_results_dir="/home/ubuntu/Desktop/autonomy_results",
        dashboard_out="/home/ubuntu/Desktop/market_dashboard.csv",
        eval_config_name="topstep_50k",
        use_online_updates=True,  # training = True
        model_save_dir="/home/ubuntu/Desktop/artifacts/symbol_models",
        num_evolution_windows=3,
        evolution_window_days=10,
        num_autonomy_windows=3,
        autonomy_window_days=10,
        baseline_dir="/home/ubuntu/Desktop/baselines",
        store_baseline=True,
        preset_iterations=20,
        quick_mode=False,
    )


def build_recent_windows(num_windows: int, window_days: int) -> list[tuple[str, str]]:
    today = date.today()
    total_span = num_windows * window_days
    start = today - timedelta(days=total_span)
    windows: list[tuple[str, str]] = []
    cur = start
    for _ in range(num_windows):
        w_start = cur
        w_end = w_start + timedelta(days=window_days)
        windows.append((w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        cur = w_end
    return windows


def build_eval_window(window_days: int = 10) -> tuple[str, str]:
    today = date.today()
    end = today - timedelta(days=1)
    start = end - timedelta(days=window_days)
    return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def get_baseline_paths(cfg: TrainAllConfig) -> tuple[str, str]:
    os.makedirs(cfg.baseline_dir, exist_ok=True)
    baseline_csv = os.path.join(cfg.baseline_dir, "dashboard_baseline.csv")
    baseline_report = os.path.join(cfg.baseline_dir, "dashboard_baseline_report.md")
    return baseline_csv, baseline_report


def get_state_path(cfg: TrainAllConfig) -> str:
    base_dir = os.path.dirname(cfg.dashboard_out) or "."
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "train_all_state.json")


def load_run_state(state_path: str) -> dict:
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_run_state(state_path: str, state: dict) -> None:
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, state_path)


def run_preset_evolution(cfg: TrainAllConfig, windows: list[tuple[str, str]]):
    print("\n[train_all] STEP 1/4: PRESET EVOLUTION", flush=True)
    print(f"[train_all] Evolving presets for symbols: {', '.join(cfg.symbols)}", flush=True)
    print(f"[train_all] Evolution windows: {windows}", flush=True)
    print("[train_all] Calling preset_evolve.evolve_presets_for_markets(...)", flush=True)
    preset_evolve.evolve_presets_for_markets(
        symbols=cfg.symbols,
        windows=windows,
        preset_dir=cfg.preset_dir,
        data_dir=cfg.data_dir,
        results_dir=cfg.evolution_results_dir,
        model_path=cfg.model_path,
        eval_config_name=cfg.eval_config_name,
        n_iterations=cfg.preset_iterations,
    )
    print("[train_all] Preset evolution completed.", flush=True)
    print(f"[train_all] Expect updated best presets in: {cfg.preset_dir}", flush=True)
    print(
        f"[train_all] Evolution summary CSV: {cfg.evolution_results_dir}/preset_evolution_market_ranking.csv",
        flush=True,
    )
    print("-" * 80, flush=True)


def run_model_training_eval(cfg: TrainAllConfig, eval_window: tuple[str, str]):
    start, end = eval_window
    print("\n[train_all] STEP 2/4: MODEL TRAINING EVAL", flush=True)
    print(f"[train_all] Eval window: {start} -> {end}", flush=True)
    print(f"[train_all] Online model updates: {cfg.use_online_updates}", flush=True)
    if cfg.use_online_updates and cfg.model_save_dir:
        print(f"[train_all] Updated models will be saved under: {cfg.model_save_dir}", flush=True)
    print("[train_all] Calling market_eval.evaluate_markets(...) with use_best_presets_dir.", flush=True)

    market_eval.evaluate_markets(
        symbols=cfg.symbols,
        start=start,
        end=end,
        model_path=cfg.model_path,
        data_dir=cfg.data_dir,
        results_dir=cfg.eval_results_dir,
        online=False,
        overwrite_data=False,
        commission=0.0,
        slip_ticks=0.0,
        preset_dir=cfg.preset_dir,
        eval_config_name=cfg.eval_config_name,
        eval_check=True,
        online_update_models=cfg.use_online_updates,
        model_save_dir=cfg.model_save_dir,
    )

    print("[train_all] market_eval completed.", flush=True)
    print(f"[train_all] Eval summary CSV: {os.path.join(cfg.eval_results_dir, 'market_eval_summary.csv')}", flush=True)
    if cfg.use_online_updates and cfg.model_save_dir:
        print("[train_all] Check symbol models under:", cfg.model_save_dir, flush=True)
    print("-" * 80, flush=True)


def run_autonomy_tests(cfg: TrainAllConfig, windows: list[tuple[str, str]]):
    print("\n[train_all] STEP 3/4: AUTONOMY TESTS", flush=True)
    print(f"[train_all] Running autonomy tests for symbols: {', '.join(cfg.symbols)}", flush=True)
    print(f"[train_all] Autonomy windows: {windows}", flush=True)
    print("[train_all] Calling autonomy_test.run_autonomy_test(...)", flush=True)
    autonomy_test.run_autonomy_test(
        symbols=cfg.symbols,
        windows=windows,
        model_path=cfg.model_path,
        preset_dir=cfg.preset_dir,
        eval_config_name=cfg.eval_config_name,
        data_dir=cfg.data_dir,
        results_dir=cfg.autonomy_results_dir,
    )
    print("[train_all] Autonomy tests completed.", flush=True)
    print(
        f"[train_all] Autonomy summary CSV: {cfg.autonomy_results_dir}/summary.csv",
        flush=True,
    )
    print("-" * 80, flush=True)


def run_market_dashboard(cfg: TrainAllConfig):
    print("\n[train_all] STEP 4/4: DASHBOARD + REPORT", flush=True)
    print("[train_all] Building consolidated dashboard...", flush=True)
    dashboard = market_dashboard.build_dashboard(
        os.path.join(cfg.eval_results_dir, "market_eval_summary.csv"),
        os.path.join(cfg.evolution_results_dir, "preset_evolution_market_ranking.csv"),
        os.path.join(cfg.autonomy_results_dir, "summary.csv"),
    )
    dashboard.to_csv(cfg.dashboard_out, index=False)
    print(f"[train_all] Dashboard written to: {cfg.dashboard_out}", flush=True)
    return dashboard


def compare_with_baseline(cfg: TrainAllConfig) -> dict:
    baseline_csv, _ = get_baseline_paths(cfg)
    if not os.path.exists(baseline_csv):
        print("[train_all] Regression check: no baseline found; this run will set the baseline.", flush=True)
        return {"has_baseline": False}

    current = pd.read_csv(cfg.dashboard_out)
    baseline = pd.read_csv(baseline_csv)
    if current.empty or baseline.empty:
        print("[train_all] Current or baseline dashboard empty; skipping regression check.", flush=True)
        return {"has_baseline": False}

    merged = current.merge(baseline, on="symbol", suffixes=("_cur", "_base"))
    rows: List[dict] = []
    for _, row in merged.iterrows():
        sym = row["symbol"]
        delta_score = row.get("score_cur", 0.0) - row.get("score_base", 0.0)
        delta_eval_win = row.get("eval_win_rate_cur", 0.0) - row.get("eval_win_rate_base", 0.0)
        delta_auto_rate = row.get("pass_rate_cur", 0.0) - row.get("pass_rate_base", 0.0)
        rows.append(
            {
                "symbol": sym,
                "delta_score": delta_score,
                "delta_eval_win_rate": delta_eval_win,
                "delta_autonomy_pass_rate": delta_auto_rate,
            }
        )
    deltas = pd.DataFrame(rows)
    worst_drop = float(deltas["delta_score"].min()) if not deltas.empty else 0.0
    summary = {
        "has_baseline": True,
        "num_symbols": len(deltas),
        "num_improved_score": int((deltas["delta_score"] > 0).sum()) if not deltas.empty else 0,
        "num_worsened_score": int((deltas["delta_score"] < 0).sum()) if not deltas.empty else 0,
        "worst_score_drop": worst_drop,
    }
    print("[train_all] Regression summary:", flush=True)
    print(f"  - Symbols improved score: {summary['num_improved_score']} / {summary['num_symbols']}", flush=True)
    print(f"  - Symbols worsened score: {summary['num_worsened_score']} / {summary['num_symbols']}", flush=True)
    print(f"  - Worst score drop: {summary['worst_score_drop']:.2f}", flush=True)
    return {"has_baseline": True, "deltas": deltas, "summary": summary}


def build_train_all_report(cfg: TrainAllConfig, regression: dict | None = None) -> str:
    df = pd.read_csv(cfg.dashboard_out)

    lines = []
    lines.append("# Train-All SIM Report\n")
    lines.append(f"Eval Config: {cfg.eval_config_name}\n")
    lines.append("## Top Markets by Evolution Score\n")

    if "score" in df.columns:
        top = df.sort_values(by="score", ascending=False).head(3)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['symbol']}: score={row['score']:.1f}, "
                f"win_rate={row.get('avg_win_rate', 0.0):.2%}, "
                f"avg_max_dd={row.get('avg_max_drawdown', 0.0):.0f}"
            )

    if "autonomy_ready" in df.columns:
        not_ready = df[df["autonomy_ready"] == False]
        if not not_ready.empty:
            lines.append("\n## Markets NOT Autonomy-Ready\n")
            for _, row in not_ready.iterrows():
                reason = row.get("autonomy_reason", "see autonomy summary")
                lines.append(f"- {row['symbol']}: {reason}")

    if "avg_max_drawdown" in df.columns:
        deep_dd = df[df["avg_max_drawdown"] < -1500]
        if not deep_dd.empty:
            lines.append("\n## Risk Flags (Deep Drawdown)\n")
            for _, row in deep_dd.iterrows():
                lines.append(f"- {row['symbol']}: avg_max_dd={row['avg_max_drawdown']:.0f}")

    lines.append("\n## Notes\n")
    lines.append("- All results are SIM ONLY. No live orders were sent.")
    lines.append("- Updated presets saved in preset_search_results/.")
    if cfg.use_online_updates and cfg.model_save_dir:
        lines.append(f"- Updated per-market models saved under {cfg.model_save_dir}.")

    if regression and regression.get("has_baseline"):
        lines.append("\n## Regression Check vs Baseline\n")
        summary = regression.get("summary", {})
        if summary:
            lines.append(
                f"- Symbols improved score: {summary.get('num_improved_score', 0)} / {summary.get('num_symbols', 0)}"
            )
            lines.append(
                f"- Symbols worsened score: {summary.get('num_worsened_score', 0)} / {summary.get('num_symbols', 0)}"
            )
            lines.append(f"- Worst score drop: {summary.get('worst_score_drop', 0.0):.1f}")
        deltas = regression.get("deltas")
        if isinstance(deltas, pd.DataFrame) and not deltas.empty:
            worst = deltas.sort_values(by="delta_score").head(2)
            for _, row in worst.iterrows():
                lines.append(
                    f"  - {row['symbol']}: delta_score={row['delta_score']:.1f}, "
                    f"delta_eval_win_rate={row['delta_eval_win_rate']:.2%}"
                )
    else:
        lines.append("\n## Regression Check\n")
        lines.append("- No baseline available yet; establish one with store_baseline.")

    return "\n".join(lines)


def save_train_all_report(cfg: TrainAllConfig, text: str):
    out_path = cfg.dashboard_out.replace(".csv", "_train_all_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[train_all] Train-all report written to: {out_path}", flush=True)
    return out_path


def maybe_update_baseline(cfg: TrainAllConfig, regression: dict, report_path: str | None = None) -> None:
    baseline_csv, baseline_report = get_baseline_paths(cfg)
    if not os.path.exists(baseline_csv):
        print("[train_all] Baseline update: ACCEPTED.", flush=True)
        print("[train_all] Stored current dashboard as new baseline.", flush=True)
        shutil.copy2(cfg.dashboard_out, baseline_csv)
        if report_path and os.path.exists(report_path):
            shutil.copy2(report_path, baseline_report)
        return

    summary = (regression or {}).get("summary")
    if not summary or not summary.get("has_baseline"):
        print("[train_all] Baseline update: skipped (regression summary unavailable).", flush=True)
        return

    if summary.get("num_worsened_score", 0) == 0:
        print("[train_all] Baseline update: ACCEPTED.", flush=True)
        print("[train_all] Stored current dashboard as new baseline.", flush=True)
        shutil.copy2(cfg.dashboard_out, baseline_csv)
        if report_path and os.path.exists(report_path):
            shutil.copy2(report_path, baseline_report)
    else:
        print("[train_all] Baseline update: REJECTED due to regressions.", flush=True)
        print("[train_all] Keeping previous baseline for safety.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train presets + models for all markets (SIM ONLY).")
    parser.add_argument("--symbols", nargs="+", help="Override default symbols list.")
    parser.add_argument("--no-online-updates", action="store_true", help="Disable online model updates.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick debug cycle with one symbol, shorter windows, and fewer preset iterations.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any saved progress and re-run all steps from scratch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = default_config()
    if args.symbols:
        cfg.symbols = [s.upper() for s in args.symbols]
    if args.no_online_updates:
        cfg.use_online_updates = False
    if args.quick:
        cfg.quick_mode = True
        cfg.symbols = [cfg.symbols[0]] if cfg.symbols else ["ES"]
        cfg.num_evolution_windows = 1
        cfg.evolution_window_days = 3
        cfg.num_autonomy_windows = 1
        cfg.autonomy_window_days = 3
        cfg.preset_iterations = 3
        print("[train_all] QUICK MODE enabled — reduced symbols/windows/iterations for faster debugging.", flush=True)

    print("=" * 80, flush=True)
    print("[train_all] SIM-ONLY TRAINING CYCLE", flush=True)
    print("[train_all] No live orders are sent. This is for SIM eval + preflight only.", flush=True)
    print("=" * 80, flush=True)
    print(f"[train_all] Using eval_config={cfg.eval_config_name}", flush=True)
    print(f"[train_all] Symbols={cfg.symbols}", flush=True)
    print(f"[train_all] Model path={cfg.model_path}", flush=True)
    print(f"[train_all] Preset dir={cfg.preset_dir}", flush=True)
    print(f"[train_all] Data dir={cfg.data_dir}", flush=True)
    print(f"[train_all] Eval results dir={cfg.eval_results_dir}", flush=True)
    print(f"[train_all] Evolution results dir={cfg.evolution_results_dir}", flush=True)
    print(f"[train_all] Autonomy results dir={cfg.autonomy_results_dir}", flush=True)
    print(f"[train_all] Dashboard path={cfg.dashboard_out}", flush=True)
    print(f"[train_all] Online model updates={cfg.use_online_updates}", flush=True)
    print(f"[train_all] Model save dir={cfg.model_save_dir}", flush=True)
    print(f"[train_all] Baseline dir={cfg.baseline_dir}", flush=True)
    print(f"[train_all] Preset iterations={cfg.preset_iterations}", flush=True)
    print("=" * 80, flush=True)

    evo_windows = build_recent_windows(cfg.num_evolution_windows, cfg.evolution_window_days)
    auto_windows = build_recent_windows(cfg.num_autonomy_windows, cfg.autonomy_window_days)
    eval_window = build_eval_window(window_days=cfg.evolution_window_days)

    print("[train_all] Evolution windows:", flush=True)
    for s, e in evo_windows:
        print(f"  - {s} -> {e}", flush=True)
    print("[train_all] Autonomy windows:", flush=True)
    for s, e in auto_windows:
        print(f"  - {s} -> {e}", flush=True)
    print(f"[train_all] Eval window: {eval_window[0]} -> {eval_window[1]}", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    state_path = get_state_path(cfg)
    if args.restart and os.path.exists(state_path):
        os.remove(state_path)
        print(f"[train_all] Cleared previous run state at {state_path}.", flush=True)
    run_state = {}
    if os.path.exists(state_path):
        run_state = load_run_state(state_path)
        print(f"[train_all] Found previous run state; resuming remaining steps. ({state_path})", flush=True)
    else:
        print(f"[train_all] No prior run state found. Fresh start. ({state_path})", flush=True)

    def step_completed(name: str) -> bool:
        return bool(run_state.get(name))

    def mark_completed(name: str) -> None:
        run_state[name] = True
        save_run_state(state_path, run_state)

    if step_completed("preset_evolution"):
        print("[train_all][RESUME] Skipping preset evolution (already completed).", flush=True)
    else:
        step_start = time.time()
        run_preset_evolution(cfg, evo_windows)
        t1 = time.time()
        mark_completed("preset_evolution")
        print(f"[train_all][TIMING] Preset evolution took {t1 - step_start:.1f}s", flush=True)

    if step_completed("model_eval"):
        print("[train_all][RESUME] Skipping model training eval (already completed).", flush=True)
    else:
        step_start = time.time()
        run_model_training_eval(cfg, eval_window)
        t2 = time.time()
        mark_completed("model_eval")
        print(f"[train_all][TIMING] Model-training eval took {t2 - step_start:.1f}s", flush=True)

    if step_completed("autonomy"):
        print("[train_all][RESUME] Skipping autonomy tests (already completed).", flush=True)
    else:
        step_start = time.time()
        run_autonomy_tests(cfg, auto_windows)
        t3 = time.time()
        mark_completed("autonomy")
        print(f"[train_all][TIMING] Autonomy tests took {t3 - step_start:.1f}s", flush=True)

    regression = None
    report_path = None

    if step_completed("dashboard"):
        print("[train_all][RESUME] Skipping dashboard/report build (already completed).", flush=True)
    else:
        step_start = time.time()
        run_market_dashboard(cfg)
        regression = compare_with_baseline(cfg)
        report = build_train_all_report(cfg, regression)
        report_path = save_train_all_report(cfg, report)
        if cfg.store_baseline:
            maybe_update_baseline(cfg, regression, report_path)
        t4 = time.time()
        mark_completed("dashboard")
        print(f"[train_all][TIMING] Dashboard+report took {t4 - step_start:.1f}s", flush=True)

    if regression is None:
        regression = compare_with_baseline(cfg)
    if report_path is None:
        report_path = cfg.dashboard_out.replace(".csv", "_train_all_report.md")

    if step_completed("journal"):
        print("[train_all][RESUME] Skipping journal coach (already completed).", flush=True)
    else:
        coach_cfg = journal_coach.default_journal_config()
        coach_cfg.symbols = cfg.symbols
        coach_cfg.trades_dir = cfg.eval_results_dir
        coach_cfg.eval_results_dir = cfg.eval_results_dir
        coach_cfg.autonomy_results_dir = cfg.autonomy_results_dir
        coach_cfg.out_dir = os.path.join(os.path.dirname(cfg.dashboard_out), "journal_reports")
        coach_start = time.time()
        try:
            print("[train_all] Running Journal Coach (SIM behavioral analysis)...", flush=True)
            journal_coach.run_journal_coach(coach_cfg)
            print("[train_all] Journal Coach completed.", flush=True)
        except Exception as exc:
            print(f"[train_all] Journal Coach failed with error: {exc!r}. Continuing run.", flush=True)
        coach_end = time.time()
        mark_completed("journal")
        print(f"[train_all][TIMING] Journal coach took {coach_end - coach_start:.1f}s", flush=True)

    if os.path.exists(state_path):
        os.remove(state_path)
        print(f"[train_all] Cleared run state at {state_path} (run completed).", flush=True)

    print(f"[train_all][TIMING] Total train_all runtime {time.time() - t0:.1f}s", flush=True)

    print("=" * 80, flush=True)
    print("[train_all] RUN COMPLETE", flush=True)
    print(f"[train_all] Dashboard: {cfg.dashboard_out}", flush=True)
    print(f"[train_all] Train-all report: {cfg.dashboard_out.replace('.csv', '_train_all_report.md')}", flush=True)
    print(f"[train_all] Baseline directory: {cfg.baseline_dir}", flush=True)
    print(
        "[train_all] Remember: all results are SIM ONLY. Use topstep_preflight and your own judgment before any real eval.",
        flush=True,
    )
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
