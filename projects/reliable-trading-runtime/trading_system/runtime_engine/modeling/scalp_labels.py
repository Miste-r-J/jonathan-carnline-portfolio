from __future__ import annotations

"""
Binary "target-first" scalper labels.

Opt-in only: existing training flows remain untouched unless callers explicitly
select the scalper label scheme.
"""

import numpy as np
import pandas as pd


def scalp_label_series(
    df: pd.DataFrame,
    target_ticks: int,
    stop_ticks: int,
    horizon_bars: int,
    tick_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_long, y_short) binary labels.

    A label is 1 if the target is hit before the stop within the forward
    ``horizon_bars`` window, otherwise 0.
    """
    df_norm = df.copy()
    df_norm.columns = [str(c).strip().lower() for c in df_norm.columns]
    required = {"high", "low", "close"}
    missing = sorted(required - set(df_norm.columns))
    if missing:
        raise KeyError(f"Missing required columns for scalp labels: {', '.join(missing)}")

    n = len(df_norm)
    y_long = np.zeros(n, dtype=np.int8)
    y_short = np.zeros(n, dtype=np.int8)
    hi = df_norm["high"].to_numpy()
    lo = df_norm["low"].to_numpy()
    close = df_norm["close"].to_numpy()

    tgt_off = float(target_ticks) * float(tick_size)
    stp_off = float(stop_ticks) * float(tick_size)

    for i in range(n):
        px = close[i]
        tgt_long, stop_long = px + tgt_off, px - stp_off
        tgt_short, stop_short = px - tgt_off, px + stp_off
        win_long = lose_long = win_short = lose_short = False
        end = min(i + 1 + int(horizon_bars), n)
        for j in range(i + 1, end):
            if hi[j] >= tgt_long:
                win_long = True
                break
            if lo[j] <= stop_long:
                lose_long = True
                break
        for j in range(i + 1, end):
            if lo[j] <= tgt_short:
                win_short = True
                break
            if hi[j] >= stop_short:
                lose_short = True
                break
        y_long[i] = 1 if (win_long and not lose_long) else 0
        y_short[i] = 1 if (win_short and not lose_short) else 0
    return y_long, y_short

