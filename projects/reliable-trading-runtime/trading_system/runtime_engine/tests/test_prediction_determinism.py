import json
import unittest

from trading_system.runtime_engine.integrations.prediction_pipeline import build_prediction_bundle, compute_features_hash


class TestPredictionDeterminism(unittest.TestCase):
    def test_prediction_bundle_hash_deterministic(self):
        features = {"f1": 1.2345, "f2": 9, "f3": "x"}
        features_hash = compute_features_hash(features)
        base = dict(
            run_id="run1",
            bar_ts="2026-01-28T14:30:00",
            instrument="ES 03-26",
            features_hash=features_hash,
            model_version="model_v1",
            proba=0.6123,
            side="LONG",
            entry_ref=4800.25,
            stop_abs=4790.0,
            target_abs=4820.0,
            size_reco=2,
            risk_meta={"p_buy": 0.6, "p_sell": 0.4},
            policy_meta_model={"preset": "test"},
        )
        b1 = build_prediction_bundle(**base)
        b2 = build_prediction_bundle(**base)
        self.assertEqual(b1["prediction_id"], b2["prediction_id"])
        self.assertEqual(b1["deterministic_hash"], b2["deterministic_hash"])

    def test_prediction_bundle_ignores_execution_noise(self):
        features = {"f1": 1.0, "f2": 2.0}
        features_hash = compute_features_hash(features)
        base = dict(
            run_id="run2",
            bar_ts="2026-01-28T14:35:00",
            instrument="ES 03-26",
            features_hash=features_hash,
            model_version="model_v2",
            proba=0.501,
            side="SHORT",
            entry_ref=4799.0,
            stop_abs=4804.0,
            target_abs=4789.0,
            size_reco=1,
            risk_meta={"p_buy": 0.6},
            policy_meta_model={"preset": "test"},
        )
        b1 = build_prediction_bundle(**base)
        exec_noise = json.dumps({"lockout": True, "position": "LONG"})
        _ = exec_noise  # execution noise should not affect bundle
        b2 = build_prediction_bundle(**base)
        self.assertEqual(b1["deterministic_hash"], b2["deterministic_hash"])


if __name__ == "__main__":
    unittest.main()
