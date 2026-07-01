from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POINT_VALUE = 50.0


def _normalize_contract(value: Any) -> str:
    text = str(value or "").upper()
    if "06-26" in text or "JUN26" in text:
        return "ES 06-26"
    if "09-26" in text or "SEP26" in text:
        return "ES 09-26"
    return text


def _load_bars(path: Path) -> pd.DataFrame:
    bars = pd.read_csv(path)
    bars["Contract"] = bars["Contract"].map(_normalize_contract)
    bars["Datetime"] = pd.to_datetime(bars["Datetime"], utc=True, errors="coerce")
    for column in ("Open", "High", "Low", "Close"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.dropna(subset=["Datetime", "High", "Low"]).sort_values(
        ["Contract", "Datetime"]
    )


def _load_trades(path: Path, *, source: str, cost_per_round_trip: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename = {
        "contract": "Contract",
        "actual_entry_price": "entry_price",
        "actual_exit_price": "exit_price",
    }
    for old, new in rename.items():
        if old in frame and new not in frame:
            frame[new] = frame[old]
    frame["Contract"] = frame.get("Contract", "").map(_normalize_contract)
    frame["entry_ts"] = pd.to_datetime(frame["entry_ts"], utc=True, errors="coerce")
    frame["exit_ts"] = pd.to_datetime(frame["exit_ts"], utc=True, errors="coerce")
    frame["entry_price"] = pd.to_numeric(frame["entry_price"], errors="coerce")
    frame["exit_price"] = pd.to_numeric(frame["exit_price"], errors="coerce")
    frame["qty"] = pd.to_numeric(
        frame["qty"] if "qty" in frame else frame.get("filled_qty", 1.0), errors="coerce"
    ).fillna(1.0)
    frame["side"] = frame["side"].astype(str).str.upper()
    frame = frame.dropna(subset=["entry_ts", "exit_ts", "entry_price", "exit_price"])
    direction = np.where(frame["side"].eq("LONG"), 1.0, -1.0)
    frame["pnl_points"] = (
        (frame["exit_price"] - frame["entry_price"]) * direction
    )
    frame["gross_pnl_usd"] = frame["pnl_points"] * frame["qty"] * POINT_VALUE
    frame["estimated_costs_usd"] = float(cost_per_round_trip) * frame["qty"]
    frame["net_pnl_usd"] = frame["gross_pnl_usd"] - frame["estimated_costs_usd"]
    frame["source"] = source
    return frame.reset_index(drop=True)


def _attach_excursions(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    result = trades.copy()
    mfe: list[float] = []
    mae: list[float] = []
    capture: list[float | None] = []
    for row in result.itertuples(index=False):
        window = bars.loc[
            bars["Contract"].eq(row.Contract)
            & bars["Datetime"].between(row.entry_ts, row.exit_ts, inclusive="both")
        ]
        if window.empty:
            mfe.append(np.nan)
            mae.append(np.nan)
            capture.append(np.nan)
            continue
        if row.side == "LONG":
            favorable = float(window["High"].max() - row.entry_price)
            adverse = float(row.entry_price - window["Low"].min())
        else:
            favorable = float(row.entry_price - window["Low"].min())
            adverse = float(window["High"].max() - row.entry_price)
        favorable = max(favorable, 0.0)
        adverse = max(adverse, 0.0)
        mfe.append(favorable)
        mae.append(adverse)
        capture.append(row.pnl_points / favorable if favorable > 0 else np.nan)
    result["mfe_points"] = mfe
    result["mae_points"] = mae
    result["mfe_capture_ratio"] = capture
    result["equity_net_usd"] = result["net_pnl_usd"].cumsum()
    running_peak = result["equity_net_usd"].cummax().clip(lower=0.0)
    result["drawdown_net_usd"] = result["equity_net_usd"] - running_peak
    return result


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    gross_wins = float(frame.loc[frame["net_pnl_usd"] > 0, "net_pnl_usd"].sum())
    gross_losses = float(frame.loc[frame["net_pnl_usd"] < 0, "net_pnl_usd"].sum())
    return {
        "trades": int(len(frame)),
        "wins": int((frame["net_pnl_usd"] > 0).sum()),
        "losses": int((frame["net_pnl_usd"] < 0).sum()),
        "win_rate": float((frame["net_pnl_usd"] > 0).mean()) if len(frame) else 0.0,
        "gross_pnl_usd": float(frame["gross_pnl_usd"].sum()),
        "estimated_costs_usd": float(frame["estimated_costs_usd"].sum()),
        "net_pnl_usd": float(frame["net_pnl_usd"].sum()),
        "profit_factor": gross_wins / abs(gross_losses) if gross_losses else None,
        "avg_trade_net_usd": float(frame["net_pnl_usd"].mean()) if len(frame) else 0.0,
        "max_drawdown_net_usd": float(frame["drawdown_net_usd"].min()) if len(frame) else 0.0,
        "mfe_points_mean": float(frame["mfe_points"].mean()),
        "mae_points_mean": float(frame["mae_points"].mean()),
        "mfe_capture_ratio_mean": float(frame["mfe_capture_ratio"].mean()),
        "long_net_pnl_usd": float(frame.loc[frame["side"].eq("LONG"), "net_pnl_usd"].sum()),
        "short_net_pnl_usd": float(frame.loc[frame["side"].eq("SHORT"), "net_pnl_usd"].sum()),
    }


def _graphs(frames: dict[str, pd.DataFrame], out_dir: Path) -> list[str]:
    paths: list[str] = []
    plt.figure(figsize=(11, 6))
    for label, frame in frames.items():
        plt.plot(np.arange(1, len(frame) + 1), frame["equity_net_usd"], label=label)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Cost-adjusted cumulative PnL")
    plt.xlabel("Trade number")
    plt.ylabel("PnL (USD)")
    plt.legend()
    plt.tight_layout()
    equity_path = out_dir / "equity_comparison.png"
    plt.savefig(equity_path, dpi=160)
    plt.close()
    paths.append(str(equity_path))

    summaries = {label: _summary(frame) for label, frame in frames.items()}
    plt.figure(figsize=(9, 5))
    labels = list(summaries)
    values = [summaries[label]["net_pnl_usd"] for label in labels]
    colors = ["#6b7280", "#dc2626", "#16a34a"]
    plt.bar(labels, values, color=colors[: len(labels)])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Net PnL comparison after estimated costs")
    plt.ylabel("PnL (USD)")
    plt.tight_layout()
    pnl_path = out_dir / "net_pnl_comparison.png"
    plt.savefig(pnl_path, dpi=160)
    plt.close()
    paths.append(str(pnl_path))

    plt.figure(figsize=(10, 6))
    for label, frame in frames.items():
        plt.scatter(
            frame["mfe_points"],
            frame["pnl_points"],
            s=18,
            alpha=0.55,
            label=label,
        )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("MFE (points)")
    plt.ylabel("Realized PnL (points)")
    plt.title("MFE capture by trade")
    plt.legend()
    plt.tight_layout()
    mfe_path = out_dir / "mfe_capture_comparison.png"
    plt.savefig(mfe_path, dpi=160)
    plt.close()
    paths.append(str(mfe_path))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--live-round-trip-cost", type=float, default=4.0)
    parser.add_argument("--replay-round-trip-cost", type=float, default=29.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bars = _load_bars(args.bars)
    frames = {
        "Actual live": _attach_excursions(
            _load_trades(
                args.live, source="actual_live", cost_per_round_trip=args.live_round_trip_cost
            ),
            bars,
        ),
        "Old replay": _attach_excursions(
            _load_trades(
                args.old, source="old_replay", cost_per_round_trip=args.replay_round_trip_cost
            ),
            bars,
        ),
        "New model replay": _attach_excursions(
            _load_trades(
                args.new, source="new_replay", cost_per_round_trip=args.replay_round_trip_cost
            ),
            bars,
        ),
    }
    for label, frame in frames.items():
        filename = label.lower().replace(" ", "_") + "_enriched.csv"
        frame.to_csv(args.out_dir / filename, index=False)
    report = {
        "summaries": {label: _summary(frame) for label, frame in frames.items()},
        "graphs": _graphs(frames, args.out_dir),
        "cost_assumptions": {
            "live_round_trip_cost_usd": args.live_round_trip_cost,
            "replay_round_trip_cost_usd": args.replay_round_trip_cost,
            "point_value_usd": POINT_VALUE,
        },
    }
    (args.out_dir / "phase2_relaunch_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
