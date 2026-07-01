from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("trading_system.runtime_engine.modeling.explain_attribution_report")
LOGGER.setLevel(logging.INFO)


def _load_explanations(path: Path) -> pd.DataFrame:
    records: List[dict] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            shap = payload.get("shap") or {}
            contributions = shap.get("contributions")
            if not contributions:
                positives = shap.get("top_positive") or []
                negatives = shap.get("top_negative") or []
                contributions = positives + negatives
            records.append(
                {
                    "event_id": payload.get("event_id"),
                    "timestamp": payload.get("timestamp"),
                    "side": payload.get("side"),
                    "prob": payload.get("prob"),
                    "grade": payload.get("grade"),
                    "shap_sum": shap.get("shap_sum"),
                    "base_value": shap.get("base_value"),
                    "class_index": shap.get("class_index"),
                    "contributions": contributions or [],
                }
            )
    if not records:
        raise RuntimeError(f"No explanations found in {path}")
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _categorize_tod(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "unknown"
    local_time = dt.time()
    open_start = time(7, 30)
    mid_start = time(9, 30)
    lunch_start = time(11, 30)
    close_time = time(13, 30)
    if open_start <= local_time < mid_start:
        return "open"
    if mid_start <= local_time < lunch_start:
        return "mid"
    if lunch_start <= local_time < close_time:
        return "lunch"
    return "after"


def _extract_feature_value(contributions: Sequence[Dict[str, float]], key: str) -> Optional[float]:
    for entry in contributions:
        if entry.get("name") == key:
            try:
                return float(entry.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def _load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Trades file {path} is empty.")
    lower_cols = {str(col).lower(): col for col in df.columns}
    pnl_col = None
    for candidate in ("pnl", "realized_pnl", "pnl_net", "pl"):
        if candidate in lower_cols:
            pnl_col = lower_cols[candidate]
            break
    if pnl_col is None:
        raise RuntimeError("Trades CSV must contain a PnL column (pnl, pnl_net, realized_pnl, or pl).")
    if "datetime" in lower_cols:
        df["datetime"] = pd.to_datetime(df[lower_cols["datetime"]], errors="coerce")
    df["event_id"] = df.get(lower_cols.get("event_id", ""), pd.NA)
    df["pnl_value"] = pd.to_numeric(df[pnl_col], errors="coerce")
    return df


def _merge_explanations(
    expl: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if "event_id" in expl.columns and "event_id" in trades.columns:
        merged = pd.merge(expl, trades, on="event_id", how="inner", suffixes=("", "_trade"))
    else:
        if "timestamp" not in expl.columns or "datetime" not in trades.columns:
            raise RuntimeError("Unable to join explanations and trades; provide event_id or matching timestamps.")
        merged = pd.merge(expl, trades, left_on="timestamp", right_on="datetime", how="inner", suffixes=("", "_trade"))
    if merged.empty:
        raise RuntimeError("No overlapping rows between explanations and trades.")
    merged["pnl_value"] = pd.to_numeric(merged["pnl_value"], errors="coerce")
    merged = merged.dropna(subset=["pnl_value"])
    merged["outcome"] = np.where(merged["pnl_value"] >= 0, "winner", "loser")
    merged["tod_bucket"] = merged["timestamp"].apply(_categorize_tod)
    merged["vol_proxy"] = merged["contributions"].apply(lambda contrib: _extract_feature_value(contrib, "vol_regime_z"))
    if merged["vol_proxy"].isna().all():
        merged["vol_proxy"] = merged["contributions"].apply(lambda contrib: _extract_feature_value(contrib, "atr_14"))
    if merged["vol_proxy"].isna().all():
        merged["vol_bucket"] = "unknown"
    else:
        values = merged["vol_proxy"].astype(float)
        z = (values - values.mean()) / (values.std() + 1e-9)
        merged["vol_bucket"] = pd.cut(
            z,
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=["low", "normal", "high"],
        ).astype(str)
    return merged


def _build_feature_long_df(merged: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in merged.iterrows():
        contributions = row.get("contributions") or []
        for entry in contributions:
            try:
                shap_val = float(entry.get("shap", 0.0))
            except (TypeError, ValueError):
                shap_val = 0.0
            rows.append(
                {
                    "event_id": row.get("event_id"),
                    "timestamp": row.get("timestamp"),
                    "feature": entry.get("name"),
                    "abs_shap": abs(shap_val),
                    "signed_shap": shap_val,
                    "outcome": row.get("outcome"),
                    "tod_bucket": row.get("tod_bucket"),
                    "vol_bucket": row.get("vol_bucket"),
                }
            )
    if not rows:
        raise RuntimeError("No feature contributions available for attribution.")
    return pd.DataFrame(rows)


def _plot_bar_chart(df: pd.DataFrame, out_path: Path, title: str, *, top_n: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    winners = df[df["outcome"] == "winner"].nlargest(top_n, "mean_abs_shap").set_index("feature")["mean_abs_shap"]
    losers = df[df["outcome"] == "loser"].nlargest(top_n, "mean_abs_shap").set_index("feature")["mean_abs_shap"]
    feature_order = list(dict.fromkeys(list(winners.index) + list(losers.index)))[:top_n]
    if not feature_order:
        LOGGER.warning("No features available for bar chart %s", title)
        return
    win_vals = [winners.get(feat, 0.0) for feat in feature_order]
    loss_vals = [losers.get(feat, 0.0) for feat in feature_order]
    indices = np.arange(len(feature_order))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.barh(indices, win_vals, height=width, label="Winners")
    plt.barh(indices + width, loss_vals, height=width, label="Losers")
    plt.yticks(indices + width / 2, feature_order)
    plt.xlabel("Mean |SHAP|")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index="tod_bucket", columns="outcome", values="abs_shap", aggfunc="mean").fillna(0.0)
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot.values, cmap="Blues", aspect="auto")
    plt.colorbar(label="Mean |SHAP|")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate winner vs loser attribution report from SHAP explanations.")
    parser.add_argument("--explanations", default="runs/live/explanations.jsonl")
    parser.add_argument("--trades", default="runs/live/trades.csv")
    parser.add_argument("--outdir", default="reports/attribution")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    explanations_path = Path(args.explanations).expanduser()
    trades_path = Path(args.trades).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading explanations from %s", explanations_path)
    expl_df = _load_explanations(explanations_path)
    LOGGER.info("Loading trades from %s", trades_path)
    trades_df = _load_trades(trades_path)
    merged = _merge_explanations(expl_df, trades_df)
    feature_long = _build_feature_long_df(merged)

    overall = (
        feature_long.groupby(["feature", "outcome"])["abs_shap"]
        .mean()
        .reset_index()
        .rename(columns={"abs_shap": "mean_abs_shap"})
    )
    top_wins = overall[overall["outcome"] == "winner"].nlargest(args.top_n, "mean_abs_shap")
    top_losses = overall[overall["outcome"] == "loser"].nlargest(args.top_n, "mean_abs_shap")

    _plot_bar_chart(overall, outdir / "winners_vs_losers.png", "Winner vs Loser Feature Contributions", top_n=args.top_n)

    tod_df = (
        feature_long.groupby(["tod_bucket", "outcome"])["abs_shap"]
        .mean()
        .reset_index()
    )
    _plot_heatmap(tod_df, outdir / "tod_heatmap.png", "Time-of-Day Mean |SHAP|")

    vol_df = (
        feature_long.groupby(["vol_bucket", "outcome"])["abs_shap"]
        .mean()
        .reset_index()
    )
    vol_pivot = vol_df.pivot_table(index="vol_bucket", columns="outcome", values="abs_shap", aggfunc="mean").fillna(0.0)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_events": int(len(merged)),
        "winners": int((merged["outcome"] == "winner").sum()),
        "losers": int((merged["outcome"] == "loser").sum()),
        "top_win_features": top_wins.to_dict(orient="records"),
        "top_loss_features": top_losses.to_dict(orient="records"),
        "tod_buckets": tod_df.to_dict(orient="records"),
        "volatility_buckets": vol_pivot.reset_index().to_dict(orient="records"),
        "plots": {
            "winners_vs_losers": str(outdir / "winners_vs_losers.png"),
            "tod_heatmap": str(outdir / "tod_heatmap.png"),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("Attribution report written to %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
