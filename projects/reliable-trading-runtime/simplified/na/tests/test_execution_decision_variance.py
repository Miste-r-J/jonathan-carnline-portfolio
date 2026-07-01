import unittest

from na.discord_addons.execution_decision import build_execution_decision
from na.discord_addons.prediction_pipeline import build_prediction_bundle, compute_features_hash


def parse_mock_execution_stream(lines):
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        out.append(line)
    return out


class TestExecutionDecisionVariance(unittest.TestCase):
    def test_execution_decision_does_not_mutate_prediction(self):
        features_hash = compute_features_hash({"f1": 1.0})
        bundle = build_prediction_bundle(
            run_id="runA",
            bar_ts="2026-01-28T14:40:00",
            instrument="ES 03-26",
            features_hash=features_hash,
            model_version="model_v1",
            proba=0.7,
            side="LONG",
            entry_ref=4801.0,
            stop_abs=4795.0,
            target_abs=4815.0,
            size_reco=1,
            risk_meta={},
            policy_meta_model={},
        )
        bundle_snapshot = dict(bundle)
        _ = build_execution_decision(
            prediction_id=bundle["prediction_id"],
            decision="block",
            reason_code="position_conflict",
            reason_detail={"pos": "LONG"},
            nt_state={"connected": True},
            lockout_state={"hard_lockout": False},
            size_final=0,
            stop_sent=None,
            target_sent=None,
            mode="paper",
            decision_ts="2026-01-28T14:40:01",
        )
        self.assertEqual(bundle_snapshot, bundle)

    def test_mock_execution_stream_parser(self):
        lines = [
            '{"type":"POSITION_SNAPSHOT","ts":"2026-01-28T14:00:00"}',
            '',
            '{"type":"FILL","ts":"2026-01-28T14:05:00"}',
        ]
        parsed = parse_mock_execution_stream(lines)
        self.assertEqual(len(parsed), 2)

    def test_block_decision_gets_default_reason_detail_when_missing(self):
        decision = build_execution_decision(
            prediction_id="pred-1",
            decision="block",
            reason_code="position_conflict",
            reason_detail=None,
            nt_state={"connected": True},
            lockout_state={"hard_lockout": False},
            size_final=0,
            stop_sent=None,
            target_sent=None,
            mode="paper",
            decision_ts="2026-01-28T14:40:01",
        )
        self.assertIn("reason_detail", decision)
        self.assertEqual(decision["reason_detail"]["reason"], "blocked_no_detail")
        self.assertEqual(decision["reason_detail"]["reason_code"], "position_conflict")


if __name__ == "__main__":
    unittest.main()
