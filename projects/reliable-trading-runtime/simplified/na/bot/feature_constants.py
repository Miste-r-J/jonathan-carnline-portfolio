# -----------------------------------------------------------------------
# feature_constants.py
# Single source of truth for all feature schema constants.
# Generated from: retrain_v2_full setup.features.json + dir.features.json
# Last updated: 2026-02-24
# DO NOT EDIT MANUALLY - regenerate via tools/gen_feature_constants.py
# -----------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Mapping

from .exceptions import ConfigurationError

PRODUCTION_TAG = "retrain_v2_full"

# -----------------------------------------------------------------------
# KNOWN TRAINING CONFIGURATION ISSUES (retrain_v2_full)
# Fix these in the NEXT retrain config before running train_phase2.py
# -----------------------------------------------------------------------
KNOWN_TRAINING_CONFIG_ISSUES: list[dict] = [
    {
        "id": "VAL_SPLIT_MISMATCH",
        "severity": "high",
        "description": (
            "Setup model val starts 2025-06-04; direction model val starts 2025-04-17. "
            "~7-week delta means val metrics are not comparable across models. "
            "Direction model had more 2025 OOS data available during threshold optimization."
        ),
        "fix": (
            "In next train_phase2.py run, set val_start to the SAME date for both models. "
            "Recommended: align both to 2025-06-01 or later."
        ),
        "affected_model": "retrain_v2_full",
        "affected_roles": ["direction"],
    },
]


# All 201 features the production model requires.
# Pipeline MUST produce every one of these. Missing = hard error.
MANDATORY_MODEL_FEATURES: list[str] = [
    "above_ema_fast",
    "above_ema_slow",
    "above_orb_flag",
    "above_orb_high",
    "above_vwap",
    "after_orb_flag",
    "aoi_prev_overnight_high",
    "aoi_prev_overnight_low",
    "aoi_prev_rth_high",
    "aoi_prev_rth_low",
    "at_any_aoi_band",
    "at_lower_aoi_band",
    "at_upper_aoi_band",
    "atr_14",
    "bb_bw_20",
    "bb_dn_20",
    "bb_up_20",
    "below_orb_flag",
    "below_orb_low",
    "body",
    "body_to_range",
    "broke_hod_last_n",
    "broke_lod_last_n",
    "cd_proxy",
    "cd_proxy_norm",
    "clv",
    "clv_smooth_5",
    "compression_z_12",
    "dist_abs_to_overnight_high",
    "dist_abs_to_overnight_low",
    "dist_abs_to_prev_rth_high",
    "dist_abs_to_prev_rth_low",
    "dist_orb_high",
    "dist_orb_high_atr",
    "dist_orb_high_ticks",
    "dist_orb_low",
    "dist_orb_low_atr",
    "dist_orb_low_ticks",
    "dist_to_hod_ticks",
    "dist_to_last_pivot_high_ticks",
    "dist_to_last_pivot_low_ticks",
    "dist_to_lod_ticks",
    "dist_to_orb_high",
    "dist_to_orb_high_atr",
    "dist_to_orb_low",
    "dist_to_orb_low_atr",
    "dist_to_overnight_high",
    "dist_to_overnight_high_atr",
    "dist_to_overnight_low",
    "dist_to_overnight_low_atr",
    "dist_to_prev_high",
    "dist_to_prev_high_atr",
    "dist_to_prev_low",
    "dist_to_prev_low_atr",
    "dist_to_sess_high",
    "dist_to_sess_low",
    "dn_count_10",
    "dn_count_20",
    "dn_count_5",
    "dow_0",
    "dow_1",
    "dow_2",
    "dow_3",
    "dow_4",
    "gap_sma_10",
    "gap_sma_20",
    "gap_sma_5",
    "gap_sma_50",
    "hi_break_follow",
    "hi_break_pullback",
    "hl_range",
    "impulse_atr",
    "in_rth",
    "inside_orb_flag",
    "is_after_orb",
    "is_inside_orb",
    "is_near_hod",
    "is_near_lod",
    "is_orb_window",
    "is_rth",
    "kc_dn_20",
    "kc_up_20",
    "lo_break_follow",
    "lo_break_pullback",
    "logret_1",
    "lower_wick",
    "lower_wick_ratio",
    "mins_from_open",
    "mins_since_orb_hi_break",
    "mins_since_orb_lo_break",
    "minutes_since_orb_end",
    "minutes_since_rth_open",
    "mom10_over_vol",
    "mom_10",
    "near_any_aoi_flag",
    "near_orb_high_flag",
    "near_orb_low_flag",
    "near_overnight_high_flag",
    "near_overnight_low_flag",
    "near_prev_high_flag",
    "near_prev_low_flag",
    "nearest_aoi_distance",
    "nearest_aoi_price",
    "nearest_aoi_type_code",
    "oc_range",
    "orb15_high",
    "orb15_low",
    "orb15_mid",
    "orb15_rng",
    "orb_aoi_bias",
    "orb_aoi_confirmation",
    "orb_aoi_direction",
    "orb_aoi_long_candidate",
    "orb_aoi_short_candidate",
    "orb_high",
    "orb_low",
    "orb_mid",
    "orb_range",
    "overnight_gap",
    "overnight_high",
    "overnight_low",
    "pivot_high_2",
    "pivot_high_3",
    "pivot_high_5",
    "pivot_low_2",
    "pivot_low_3",
    "pivot_low_5",
    "pressure_10",
    "pressure_10_vs_50",
    "pressure_20",
    "pressure_5",
    "pressure_50",
    "pressure_5_vs_20",
    "prev_session_high",
    "prev_session_low",
    "price_minus_orb_high",
    "price_minus_orb_low",
    "range",
    "regime_compression_score",
    "regime_ema_slope",
    "regime_ema_spread",
    "regime_id",
    "regime_range_expansion",
    "regime_time_close",
    "regime_time_mid",
    "regime_time_open",
    "regime_vol_z",
    "regime_vwap_slope",
    "ret_1",
    "ret_10",
    "ret_5",
    "rexp_over_atr",
    "rsi_14",
    "rsx_14",
    "setup_hod_retest_long",
    "setup_lod_retest_short",
    "setup_orb_breakout_long",
    "setup_orb_breakout_short",
    "setup_present",
    "setup_vwap_reclaim_long",
    "setup_vwap_reject_short",
    "signed_vol",
    "signed_vol_sum_10",
    "signed_vol_sum_20",
    "signed_vol_sum_5",
    "signed_vol_sum_50",
    "slope_ema_20",
    "slope_ema_50",
    "slope_ema_9",
    "slope_vwap_20",
    "streak_dn",
    "streak_up",
    "trend_alignment_long",
    "trend_alignment_short",
    "trend_bias_10",
    "trend_bias_20",
    "trend_bias_5",
    "true_range",
    "up_count_10",
    "up_count_20",
    "up_count_5",
    "upper_wick",
    "upper_wick_ratio",
    "vol_per_range",
    "vol_realized_10",
    "vol_realized_5",
    "vol_regime",
    "vol_regime_bin",
    "vol_regime_z",
    "vol_rel",
    "vol_spike",
    "vol_sum_10",
    "vol_sum_20",
    "vol_sum_5",
    "vol_sum_50",
    "vol_z",
    "vwap_cross",
    "vwap_cross_streak",
    "vwap_sess",
    "wick_reject_lower",
    "wick_reject_upper",
]


# Features that are safe to zero-fill if absent (non-model infrastructure cols).
# These are NOT in MANDATORY_MODEL_FEATURES.
SAFE_ZERO_FILL_FEATURES: list[str] = []


# Features confirmed redundant in the current model.
# Kept in pipeline to satisfy the frozen model schema.
# Remove from training data in the next retrain.
NEXT_RETRAIN_REMOVALS: list[str] = [
    # Duplicate/alias columns in the frozen 201-feature schema.
    "inside_orb_flag",
    "dist_orb_high_atr",
    "dist_orb_low_atr",

    # ORB naming duplication (same conceptual 15m ORB window).
    "orb_high",
    "orb_low",
    "orb_mid",
    "orb_range",

    # Redundant distance representations (pct/raw/ticks).
    "dist_orb_high",
    "dist_to_orb_high",
    "dist_orb_high_ticks",
    "dist_orb_low",
    "dist_to_orb_low",
    "dist_orb_low_ticks",

    # Redundant aggregate setup flag.
    "setup_present",
]


def validate_runtime_config_vs_model(config: Mapping[str, Any]) -> None:
    """
    Call at inference engine startup. Raises ConfigurationError if the runtime
    config would disable features required by the production model.
    """

    flow_cfg = config.get("flow_ohlcv") if isinstance(config, Mapping) else None
    flow_enabled = True
    if isinstance(flow_cfg, Mapping):
        flow_enabled = bool(flow_cfg.get("enabled", True))
    elif isinstance(flow_cfg, bool):
        flow_enabled = bool(flow_cfg)

    flow_feature_prefixes = (
        "pressure_",
        "signed_vol",
        "up_count_",
        "dn_count_",
        "trend_bias_",
        "wick_",
        "logret_",
    )
    needs_flow = any(name.startswith(flow_feature_prefixes) for name in MANDATORY_MODEL_FEATURES)
    if needs_flow and not flow_enabled:
        raise ConfigurationError(
            f"flow_ohlcv.enabled=false in config, but production model ({PRODUCTION_TAG}) "
            "requires flow_ohlcv features. Set flow_ohlcv.enabled=true."
        )
