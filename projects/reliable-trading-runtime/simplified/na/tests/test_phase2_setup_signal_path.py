import numpy as np
import pandas as pd

from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer


def _mk_streamer():
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    s._phase = "LIVE"
    s.phase2_setup_model_path = "setup.joblib"
    s.phase2_setup_model_sha = "abc123"
    s.phase2_setup_expected_features = ["f1", "f2"]
    s._phase2_setup_prob_window = []
    s._phase2_setup_prob_window_size = 16
    s._phase2_setup_prob_std_epsilon = 1e-4
    s._phase2_setup_degenerate_active = False
    s._phase2_setup_degenerate_count = 0
    s._phase2_setup_last_prob = None
    s._phase2_setup_last_window_std = None
    s._phase2_setup_last_block_reason = None
    s._log_exec_event = lambda payload: None
    return s


def test_setup_feature_validation_rejects_non_finite():
    s = _mk_streamer()
    X = pd.DataFrame({"f1": [1.0, np.nan], "f2": [2.0, 3.0]})
    ok, detail = s._validate_phase2_setup_inputs(X)
    assert ok is False
    assert detail["reason"] == "non_finite_setup_features"


def test_setup_prob_contract_rejects_len_mismatch():
    s = _mk_streamer()
    ok, detail = s._validate_phase2_setup_probs(np.array([0.1, 0.2]), expected_len=3)
    assert ok is False
    assert detail["reason"] == "setup_prob_len_mismatch"


def test_setup_degeneracy_detects_flat_setup_vs_varying_direction():
    s = _mk_streamer()
    setup_arr = np.array([0.007777777] * 12, dtype=float)
    dir_arr = np.array([0.2, 0.8, 0.3, 0.7, 0.25, 0.75, 0.35, 0.65, 0.4, 0.6, 0.1, 0.9], dtype=float)
    detail = s._track_phase2_setup_degeneracy(setup_arr, dir_arr)
    assert detail is not None
    assert detail["event"] == "setup_signal_degenerate"
    assert s._phase2_setup_degenerate_active is True
    assert int(s._phase2_setup_degenerate_count) >= 1


def test_fail_closed_phase2_uses_last_setup_block_reason():
    s = _mk_streamer()
    s._phase2_setup_last_block_reason = "setup_prob_contract_violation"
    feats = pd.DataFrame({"phase2_dir_prob_raw": [0.3, 0.6]})
    out = s._fail_closed_phase2(feats)
    assert out[0].reason == "setup_prob_contract_violation"
    assert out[1].reason == "setup_prob_contract_violation"
