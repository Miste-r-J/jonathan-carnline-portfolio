import unittest

from na.discord_addons.prediction_pipeline import build_prediction_bundle, compute_features_hash
from na.discord_addons.execution_decision import build_execution_decision


class TestLockoutDoesNotChangePrediction(unittest.TestCase):
    def test_lockout_only_affects_decision(self):
        features_hash = compute_features_hash({"f1": 2.0, "f2": 3.0})
        bundle = build_prediction_bundle(
            run_id="runL",
            bar_ts="2026-01-28T14:45:00",
            instrument="ES 03-26",
            features_hash=features_hash,
            model_version="model_v1",
            proba=0.55,
            side="LONG",
            entry_ref=4802.0,
            stop_abs=4796.0,
            target_abs=4812.0,
            size_reco=1,
            risk_meta={},
            policy_meta_model={},
        )
        decision = build_execution_decision(
            prediction_id=bundle["prediction_id"],
            decision="block",
            reason_code="lockout",
            reason_detail={"lockout": True},
            nt_state={"connected": True},
            lockout_state={"hard_lockout": True},
            size_final=0,
            stop_sent=None,
            target_sent=None,
            mode="paper",
            decision_ts="2026-01-28T14:45:01",
        )
        self.assertEqual(bundle["prediction_id"], decision["prediction_id"])


if __name__ == "__main__":
    unittest.main()
