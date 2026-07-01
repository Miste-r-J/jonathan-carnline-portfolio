import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_state(state_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        state_path,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df.groupby("datetime", as_index=False).last()


def _load_signals(signals_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        signals_path,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
    return df


def _summary(series: pd.Series) -> dict:
    clean = series.dropna()
    if clean.empty:
        return {"min": np.nan, "mean": np.nan, "median": np.nan, "max": np.nan, "std": np.nan}
    return {
        "min": clean.min(),
        "mean": clean.mean(),
        "median": clean.median(),
        "max": clean.max(),
        "std": clean.std(ddof=0),
    }


def _compute_trade_durations(signals: pd.DataFrame) -> pd.Series:
    if signals.empty or "datetime" not in signals.columns:
        return pd.Series(dtype=float)

    open_time = None
    durations = []
    for _, row in signals.iterrows():
        ts = row["datetime"]
        sig_type = row.get("type")
        if sig_type in {"OPEN", "FLIP"}:
            if open_time is not None:
                durations.append(ts - open_time)
            open_time = ts
        elif sig_type == "CLOSE" and open_time is not None:
            durations.append(ts - open_time)
            open_time = None
    if not durations:
        return pd.Series(dtype=float)
    hours = [d.total_seconds() / 3600.0 for d in durations if d.total_seconds() >= 0]
    return pd.Series(hours, dtype=float)


def analyze_run(run_dir: Path) -> None:
    state_path = run_dir / "state.csv"
    signals_path = run_dir / "signals.csv"
    if not state_path.exists():
        raise SystemExit(f"Missing state.csv at {state_path}")
    if not signals_path.exists():
        raise SystemExit(f"Missing signals.csv at {signals_path}")

    state = _load_state(state_path)
    signals = _load_signals(signals_path)

    total_bars = len(state)
    phase2_enabled_rate = state["phase2_enabled"].mean()
    phase2_pass = state["phase2_setup_pass"]
    phase2_pass_rate = phase2_pass.dropna().astype(float).mean()

    raw = state["dir_prob_raw"]
    eff = state["dir_prob_effective"]
    suppressed = np.logical_and(np.isclose(eff, 0.5), ~np.isclose(raw, 0.5))
    suppressed_rate = suppressed.mean()

    reason_counts = (
        state["phase2_reason"]
        .fillna("")
        .value_counts()
        .head(10)
        .reset_index(name="count")
        .rename(columns={"index": "phase2_reason"})
    )

    prob_table = pd.DataFrame(
        {
            "phase2_setup_prob": _summary(state["phase2_setup_prob"]),
            "dir_prob_raw": _summary(raw),
            "dir_prob_effective": _summary(eff),
        }
    ).T

    corr = np.nan
    if "phase2_setup_prob" in state.columns:
        mask = (~state["phase2_setup_prob"].isna()) & (~raw.isna())
        if mask.sum() > 1:
            deviation = (raw - 0.5).abs()
            corr = np.corrcoef(state.loc[mask, "phase2_setup_prob"], deviation[mask])[0, 1]

    signal_counts = signals["type"].value_counts().reindex(["OPEN", "FLIP", "CLOSE"]).fillna(0).astype(int)
    durations = _compute_trade_durations(signals)
    trades = len(durations)
    hold_stats = {
        "mean_hours": durations.mean(),
        "median_hours": durations.median(),
        "quantile_50": durations.quantile(0.50),
        "quantile_75": durations.quantile(0.75),
        "quantile_90": durations.quantile(0.90),
        "quantile_95": durations.quantile(0.95),
    }

    gates_sample = state[
        [
            "datetime",
            "gates",
            "gate_vwap_enabled",
            "gate_vwap_raw_pass",
            "gate_vwap_effective_pass",
            "gate_ema_enabled",
            "gate_ema_raw_pass",
            "gate_ema_effective_pass",
            "gate_tod_enabled",
            "gate_tod_raw_pass",
            "gate_tod_effective_pass",
        ]
    ].head(10)

    print("=== A) Coverage & Gating Rates ===")
    print(
        pd.DataFrame(
            [
                {"metric": "total_bars", "value": total_bars},
                {"metric": "% phase2_enabled", "value": phase2_enabled_rate},
                {"metric": "% phase2_setup_pass", "value": phase2_pass_rate},
                {"metric": "% suppressed_by_phase2", "value": suppressed_rate},
            ]
        ).to_string(index=False)
    )
    print("\nPhase2 reasons (top 10):")
    print(reason_counts.to_string(index=False))

    print("\n=== B) Probability Behavior ===")
    print(prob_table.to_string(float_format=lambda x: f"{x:0.6f}"))
    print(f"\nCorrelation phase2_setup_prob vs |dir_prob_raw-0.5|: {corr}")

    print("\n=== C) Trade / Hold Behavior ===")
    trade_table = pd.DataFrame(
        [
            {"metric": "OPEN", "value": signal_counts.get("OPEN", 0)},
            {"metric": "FLIP", "value": signal_counts.get("FLIP", 0)},
            {"metric": "CLOSE", "value": signal_counts.get("CLOSE", 0)},
            {"metric": "trades_estimated", "value": trades},
            {"metric": "hold_mean_hours", "value": hold_stats["mean_hours"]},
            {"metric": "hold_median_hours", "value": hold_stats["median_hours"]},
            {"metric": "hold_q50_hours", "value": hold_stats["quantile_50"]},
            {"metric": "hold_q75_hours", "value": hold_stats["quantile_75"]},
            {"metric": "hold_q90_hours", "value": hold_stats["quantile_90"]},
            {"metric": "hold_q95_hours", "value": hold_stats["quantile_95"]},
        ]
    )
    print(trade_table.to_string(index=False))

    print("\n=== D) Gates Sample ===")
    print(gates_sample.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a Phase-2 run directory.")
    parser.add_argument("--run_dir", required=True, help="Path to the run output directory.")
    args = parser.parse_args()
    analyze_run(Path(args.run_dir))


if __name__ == "__main__":
    main()
