from __future__ import annotations

import argparse

import pandas as pd
import pytest

from trading_system.runtime_engine.modeling.phase2_sim import Phase2DecisionPolicy, phase2_decisions
from trading_system.runtime_engine.integrations.cli.live_trading_runtime import (
    EntryPolicyConfig,
    LiveCSVStreamer,
    _approved_preset_names_for_exec_policy,
    _live_locked_preset_names,
    _live_threshold_sources_allowed,
    _resolve_probability_thresholds,
)


def test_resolve_probability_thresholds_derives_short_cut_from_short_threshold() -> None:
    args = argparse.Namespace(
        phase2_manifest_thresholds={"p_setup": 0.35, "p_long": 0.60, "p_short": 0.60},
        phase2_use_manifest_thresholds=True,
        p_setup=None,
        p_long=None,
        p_short=None,
        p_buy=None,
        p_sell=None,
    )

    resolved = _resolve_probability_thresholds(
        args=args,
        argv_tokens=[],
        preset_payload={},
        config_values=None,
    )

    assert resolved["p_setup_required"] == 0.35
    assert resolved["p_long_required"] == 0.60
    assert resolved["p_short_required"] == 0.60
    assert resolved["short_cut"] == 0.40


def test_resolve_probability_thresholds_prefers_cli_over_manifest() -> None:
    args = argparse.Namespace(
        phase2_manifest_thresholds={"p_setup": 0.35, "p_long": 0.60, "p_short": 0.60},
        phase2_use_manifest_thresholds=True,
        p_setup=0.30,
        p_long=0.55,
        p_short=0.55,
        p_buy=None,
        p_sell=None,
    )

    resolved = _resolve_probability_thresholds(
        args=args,
        argv_tokens=["--p_setup", "0.30", "--p_long", "0.55", "--p_short", "0.55"],
        preset_payload={},
        config_values=None,
    )

    assert resolved["p_setup_required"] == 0.30
    assert resolved["p_long_required"] == 0.55
    assert resolved["p_short_required"] == 0.55
    assert resolved["short_cut"] == pytest.approx(0.45)


def test_resolve_probability_thresholds_treats_cli_p_sell_as_short_cut() -> None:
    args = argparse.Namespace(
        phase2_manifest_thresholds={},
        phase2_use_manifest_thresholds=True,
        p_setup=None,
        p_long=None,
        p_short=None,
        p_buy=None,
        p_sell=0.30,
    )

    resolved = _resolve_probability_thresholds(
        args=args,
        argv_tokens=["--p_sell", "0.30"],
        preset_payload={},
        config_values=None,
    )

    assert resolved["p_short_required"] == pytest.approx(0.70)
    assert resolved["short_cut"] == pytest.approx(0.30)
    assert resolved["explicit_threshold_override"] is True


def test_resolve_probability_thresholds_reports_manifest_availability() -> None:
    args = argparse.Namespace(
        phase2_manifest_thresholds={"p_setup": 0.35, "p_long": 0.60, "p_short": 0.60},
        phase2_use_manifest_thresholds=True,
        p_setup=None,
        p_long=None,
        p_short=None,
        p_buy=None,
        p_sell=None,
    )

    resolved = _resolve_probability_thresholds(
        args=args,
        argv_tokens=[],
        preset_payload={},
        config_values=None,
    )

    assert resolved["manifest_thresholds_available"] is True
    assert resolved["explicit_threshold_override"] is False


def test_resolve_probability_thresholds_safe_preset_prefers_manifest_by_default() -> None:
    args = argparse.Namespace(
        phase2_manifest_thresholds={"p_setup": 0.35, "p_long": 0.60, "p_short": 0.60},
        phase2_use_manifest_thresholds=True,
        p_setup=None,
        p_long=None,
        p_short=None,
        p_buy=None,
        p_sell=None,
    )

    resolved = _resolve_probability_thresholds(
        args=args,
        argv_tokens=[],
        preset_payload={"p_setup": 0.44, "p_long": 0.72, "p_short": 0.72, "p_buy": 0.72, "p_sell": 0.28},
        preset_name="es_maxpack_10_full_send_prop_safe_pnl",
        config_values=None,
    )

    assert resolved["p_setup_required"] == pytest.approx(0.35)
    assert resolved["p_long_required"] == pytest.approx(0.60)
    assert resolved["p_short_required"] == pytest.approx(0.60)
    assert resolved["source_map"]["p_setup_required"] == "phase2_tag"
    assert resolved["source_map"]["p_long_required"] == "phase2_tag"
    assert resolved["source_map"]["p_short_required"] == "phase2_tag"
    assert resolved["manifest_thresholds_used"] is True


def test_resolve_probability_thresholds_high_aggressive_preset_prefers_manifest_by_default() -> None:
    args = argparse.Namespace(
        phase2_manifest_thresholds={"p_setup": 0.35, "p_long": 0.60, "p_short": 0.60},
        phase2_use_manifest_thresholds=True,
        p_setup=None,
        p_long=None,
        p_short=None,
        p_buy=None,
        p_sell=None,
    )

    resolved = _resolve_probability_thresholds(
        args=args,
        argv_tokens=[],
        preset_payload={"p_setup": 0.30, "p_long": 0.55, "p_short": 0.55, "p_buy": 0.55, "p_sell": 0.45},
        preset_name="es_maxpack_10_full_send_prop_high_aggressive_stable",
        config_values=None,
    )

    assert resolved["p_setup_required"] == pytest.approx(0.35)
    assert resolved["p_long_required"] == pytest.approx(0.60)
    assert resolved["p_short_required"] == pytest.approx(0.60)
    assert resolved["source_map"]["p_setup_required"] == "phase2_tag"
    assert resolved["source_map"]["p_long_required"] == "phase2_tag"
    assert resolved["source_map"]["p_short_required"] == "phase2_tag"
    assert resolved["manifest_thresholds_used"] is True


def test_phase2_decisions_blocks_short_when_trend_score_is_above_threshold() -> None:
    feats = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2026-04-21T10:00:00-06:00", "2026-04-21T10:05:00-06:00"]),
            "trend_score": [0.636, 0.40],
        }
    )

    result = phase2_decisions(
        feats,
        setup_probs=[0.90, 0.90],
        dir_probs=[0.30, 0.30],
        thresholds={"p_setup": 0.35, "p_long": 0.60, "p_short": 0.60},
        policy=Phase2DecisionPolicy(block_short_above_trend_score=0.55),
    )

    assert int(result.loc[0, "phase2_direction_signal"]) == 0
    assert result.loc[0, "phase2_reason"] == "trend_score_short_block"
    assert int(result.loc[1, "phase2_direction_signal"]) == -1


def test_live_threshold_sources_allow_manifest_defaults() -> None:
    assert _live_threshold_sources_allowed(
        "es_maxpack_10_full_send_prop_safe_pnl",
        {
            "p_setup_required": "phase2_tag",
            "p_long_required": "phase2_tag",
            "p_short_required": "phase2_tag",
        },
    )


def test_live_threshold_sources_allow_approved_challenge_cli_override() -> None:
    assert _live_threshold_sources_allowed(
        "es_maxpack_10_full_send_prop_challenge_community",
        {
            "p_setup_required": "cli --p_setup",
            "p_long_required": "cli --p_long",
            "p_short_required": "cli --p_short",
        },
    )


def test_live_threshold_sources_allow_safe_preset_override_mapping() -> None:
    assert _live_threshold_sources_allowed(
        "es_maxpack_10_full_send_prop_safe_pnl",
        {
            "p_setup_required": "preset p_setup",
            "p_long_required": "preset p_long",
            "p_short_required": "preset p_short",
        },
    )


def test_live_threshold_sources_allow_high_aggressive_preset_override_mapping() -> None:
    assert _live_threshold_sources_allowed(
        "es_maxpack_10_full_send_prop_high_aggressive_stable",
        {
            "p_setup_required": "preset p_setup",
            "p_long_required": "preset p_long",
            "p_short_required": "preset p_short",
        },
    )


def test_live_locked_preset_names_defaults_include_challenge_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NA_PRODUCTION_PRESET", raising=False)
    locked = _live_locked_preset_names()
    assert "es_maxpack_10_full_send_prop_safe_pnl" in locked
    assert "es_maxpack_10_full_send_prop_challenge_community" in locked
    assert "es_maxpack_10_full_send_prop_high_aggressive_stable" in locked


def test_paper_candidate_is_allowed_only_for_paper_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NA_PRODUCTION_PRESET", raising=False)
    paper = _approved_preset_names_for_exec_policy("paper")
    live = _approved_preset_names_for_exec_policy("live")
    assert "es_modelrun77_prop_v3_paper" in paper
    assert "es_modelrun77_prop_v3_paper" not in live


def test_entry_policy_config_accepts_countertrend_fields() -> None:
    cfg = EntryPolicyConfig(
        allow_countertrend_fade_in_trend=True,
        countertrend_fade_min_vwap_extension_pts=8.0,
        countertrend_fade_prob_threshold=0.30,
        countertrend_fade_size_multiplier=0.50,
        countertrend_fade_pnl_giveback_activate_r=0.75,
        countertrend_fade_pnl_giveback_close_r=0.30,
        countertrend_fade_pnl_stall_bars=2,
        countertrend_fade_pnl_stall_min_mfe_r=0.15,
        countertrend_fade_pnl_stall_close_below_r=-0.02,
        countertrend_fade_pnl_severe_adverse_r=0.60,
    )

    assert cfg.allow_countertrend_fade_in_trend is True
    assert cfg.countertrend_fade_min_vwap_extension_pts == pytest.approx(8.0)
    assert cfg.countertrend_fade_prob_threshold == pytest.approx(0.30)


def test_phase2_decisions_require_symmetric_short_confidence() -> None:
    feats = pd.DataFrame({"Datetime": pd.to_datetime(["2026-04-21T10:00:00-06:00", "2026-04-21T10:05:00-06:00"])})

    result = phase2_decisions(
        feats,
        setup_probs=[0.90, 0.90],
        dir_probs=[0.39, 0.41],
        thresholds={"p_setup": 0.35, "p_long": 0.60, "p_short": 0.60},
    )

    assert int(result.loc[0, "phase2_direction_signal"]) == -1
    assert int(result.loc[1, "phase2_direction_signal"]) == 0


def test_phase2_decisions_block_countertrend_short_only_when_strong_trend_and_weak_setup() -> None:
    feats = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2026-04-21T10:00:00-06:00", "2026-04-21T10:05:00-06:00"]),
            "Close": [5010.0, 5010.0],
            "vwap_sess": [5000.0, 5000.0],
            "ema_20": [5008.0, 5008.0],
            "ema_50": [5002.0, 5002.0],
            "trend_score": [1.55, 1.55],
        }
    )

    result = phase2_decisions(
        feats,
        setup_probs=[0.08, 0.14],
        dir_probs=[0.16, 0.16],
        thresholds={"p_setup": 0.35, "p_long": 0.60, "p_short": 0.60},
        policy=Phase2DecisionPolicy(
            countertrend_short_max_trend_score=1.40,
            countertrend_short_min_setup_when_strong_trend=0.10,
        ),
    )

    assert int(result.loc[0, "phase2_direction_signal"]) == 0
    assert int(result.loc[0, "phase2_force_open_direction_signal"]) == 0
    assert result.loc[0, "phase2_force_open_policy_reason"] == "countertrend_short_setup_filter"
    assert bool(result.loc[0, "phase2_force_open_policy_suppressed"]) is True
    assert int(result.loc[1, "phase2_direction_signal"]) == 0
    assert int(result.loc[1, "phase2_force_open_direction_signal"]) == -1


def test_phase2_decisions_block_short_flip_after_recent_long_lineage() -> None:
    feats = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                [
                    "2026-04-21T10:00:00-06:00",
                    "2026-04-21T10:05:00-06:00",
                    "2026-04-21T10:10:00-06:00",
                    "2026-04-21T10:15:00-06:00",
                ]
            )
        }
    )

    result = phase2_decisions(
        feats,
        setup_probs=[0.90, 0.90, 0.90, 0.90],
        dir_probs=[0.80, 0.79, 0.20, 0.20],
        thresholds={"p_setup": 0.35, "p_long": 0.60, "p_short": 0.60},
        policy=Phase2DecisionPolicy(short_flip_cooldown_bars_after_long_lineage=2),
    )

    assert [int(v) for v in result["phase2_force_open_direction_signal"]] == [1, 1, 0, 0]
    assert result.loc[2, "phase2_force_open_policy_reason"] == "short_flip_cooldown"
    assert result.loc[3, "phase2_force_open_policy_reason"] == "short_flip_cooldown"


def test_phase2_short_entry_block_reason_honors_policy_suppression_from_row() -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.phase2_decision_policy = Phase2DecisionPolicy()

    row = pd.Series(
        {
            "phase2_force_open_policy_reason": "short_flip_cooldown",
            "phase2_force_open_policy_suppressed": True,
            "trend_score": 0.75,
        }
    )

    reason = streamer._phase2_short_entry_block_reason("SHORT", row=row, phase2_meta=None)

    assert reason == "short_flip_cooldown"
