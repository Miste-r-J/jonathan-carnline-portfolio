from __future__ import annotations

"""
Tick/quote normalization helpers.
Schema: timestamp, symbol, bid, ask, bid_size, ask_size, last_price, last_size.
"""

import pandas as pd
from typing import Optional


REQUIRED_COLS = ["timestamp", "symbol", "bid", "ask", "bid_size", "ask_size", "last_price", "last_size"]


def normalize_tick_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    work = df.copy()
    if "symbol" not in work.columns:
        work["symbol"] = symbol
    for col in REQUIRED_COLS:
        if col not in work.columns:
            work[col] = pd.NA
    # sort and coerce timestamp
    if not pd.api.types.is_datetime64_any_dtype(work["timestamp"]):
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.sort_values("timestamp").reset_index(drop=True)
    # basic forward fill for quotes, last trade price
    for col in ["bid", "ask", "bid_size", "ask_size", "last_price", "last_size"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[["bid", "ask", "bid_size", "ask_size", "last_price", "last_size"]] = work[
        ["bid", "ask", "bid_size", "ask_size", "last_price", "last_size"]
    ].fillna(method="ffill")
    work[["bid", "ask", "bid_size", "ask_size", "last_price", "last_size"]] = work[
        ["bid", "ask", "bid_size", "ask_size", "last_price", "last_size"]
    ].fillna(0.0)
    return work[REQUIRED_COLS]


def load_tick_data(path: str, symbol: str, *, fmt: Optional[str] = None) -> pd.DataFrame:
    if fmt is None:
        fmt = "csv" if path.lower().endswith(".csv") else "parquet"
    if fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return normalize_tick_df(df, symbol)


__all__ = ["normalize_tick_df", "load_tick_data", "REQUIRED_COLS"]
