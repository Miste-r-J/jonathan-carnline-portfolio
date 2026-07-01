import unittest

from trading_system.runtime_engine.integrations.prediction_pipeline import build_prediction_bundle, compute_features_hash
from trading_system.runtime_engine.integrations.execution_decision import build_execution_decision


class TestPositionConflictBlocksOnly(unittest.TestCase):
    def test_position_conflict_block(self):
        features_hash = compute_features_hash({"f1": 1.0})
        bundle = build_prediction_bundle(
            run_id="runP",
            bar_ts="2026-01-28T14:50:00",
            instrument="ES 03-26",
            features_hash=features_hash,
            model_version="model_v1",
            proba=0.65,
            side="LONG",
            entry_ref=4803.0,
            stop_abs=4797.0,
            target_abs=4813.0,
            size_reco=2,
            risk_meta={},
            policy_meta_model={},
        )
        decision = build_execution_decision(
            prediction_id=bundle["prediction_id"],
            decision="block",
            reason_code="position_conflict",
            reason_detail={"pos_qty": 1},
            nt_state={"connected": True},
            lockout_state={"hard_lockout": False},
            size_final=0,
            stop_sent=None,
            target_sent=None,
            mode="paper",
            decision_ts="2026-01-28T14:50:01",
        )
        self.assertEqual(decision["decision"], "block")
        self.assertEqual(decision["prediction_id"], bundle["prediction_id"])


if __name__ == "__main__":
    unittest.main()
