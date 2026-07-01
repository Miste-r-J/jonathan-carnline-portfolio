import pandas as pd

from simplified.na.bot.setups_es import SetupParams, compute_es_setups


def _build_sample_df():
    idx = pd.date_range("2024-01-02 07:30", periods=10, freq="5min", tz="America/Denver")
    prices = [
        (100.0, 100.2, 99.8, 100.0),
        (100.0, 100.3, 99.9, 100.1),
        (100.1, 100.4, 99.9, 100.2),
        (100.2, 100.8, 100.1, 100.75),  # ORB breakout
        (100.6, 100.9, 100.4, 100.85),
        (100.4, 100.6, 100.2, 100.45),  # Retest near HOD
        (100.3, 100.4, 100.0, 100.1),
        (99.9, 100.1, 99.6, 99.8),
        (99.7, 99.8, 99.4, 99.5),
        (99.5, 99.6, 99.2, 99.3),
    ]
    frame = pd.DataFrame(prices, columns=["Open", "High", "Low", "Close"], index=idx)
    frame[f"orb15_high"] = 100.2
    frame[f"orb15_low"] = 99.8
    frame["vwap_sess"] = pd.Series([100.0 + i * 0.02 for i in range(len(frame))], index=idx)
    frame["atr_14"] = 0.4
    return frame


def test_compute_es_setups_flags_orb_and_hod_retest():
    df = _build_sample_df()
    params = SetupParams(confirm_ticks=2, retest_max_dist_ticks=2, retest_lookahead_bars=3)

    setups = compute_es_setups(df, tick_size=0.25, params=params)

    # ORB breakout detected on bar 3
    assert setups["setup_orb_breakout_long"].iloc[3] == 1
    assert setups["setup_present"].iloc[3] == 1

    # Retest detected on bar 5 (near HOD)
    assert setups["setup_hod_retest_long"].iloc[5] == 1

    # Structure features should provide distances and pivots without NaN
    assert pd.notna(setups["dist_orb_high_ticks"].iloc[5])
    assert pd.notna(setups["dist_to_hod_ticks"].iloc[5])
    assert "pivot_high_5" in setups.columns
    assert pd.notna(setups["compression_z_12"].iloc[-1])
