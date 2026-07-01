import json
import unittest

from reliable_event_bridge import Command, ExecutionReceipt, ReliableEventBridge


class ReliableEventBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = ReliableEventBridge(max_queue=2)
        self.generation = self.bridge.connect(session_nonce="session-a")

    def command(self, command_id: str = "cmd-001", *, generation: int | None = None) -> Command:
        return Command(
            command_id=command_id,
            generation=self.generation if generation is None else generation,
            action="APPLY_CHANGE",
            payload={"resource": "example"},
        )

    def make_ready(self) -> None:
        accepted = self.bridge.acknowledge_handshake(
            generation=self.generation,
            session_nonce="session-a",
        )
        self.assertTrue(accepted)

    def test_command_is_rejected_before_handshake(self) -> None:
        self.assertEqual("rejected", self.bridge.submit(self.command()))
        self.assertEqual("handshake_not_ready", self.bridge.ledger[-1]["reason"])

    def test_handshake_rejects_wrong_connection_identity(self) -> None:
        accepted = self.bridge.acknowledge_handshake(
            generation=self.generation,
            session_nonce="wrong-session",
        )
        self.assertFalse(accepted)
        self.assertTrue(self.bridge.degraded)

    def test_stale_generation_is_rejected(self) -> None:
        self.make_ready()
        self.assertEqual(
            "rejected",
            self.bridge.submit(self.command(generation=self.generation - 1)),
        )
        self.assertEqual("stale_connection_generation", self.bridge.ledger[-1]["reason"])

    def test_reconnect_records_abandoned_queued_work(self) -> None:
        self.make_ready()
        self.bridge.submit(self.command())
        next_generation = self.bridge.connect(session_nonce="session-b")
        self.assertEqual(self.generation + 1, next_generation)
        self.assertEqual(0, self.bridge.queue_depth)
        events = [record["event"] for record in self.bridge.ledger]
        self.assertIn("connection_reset_abandoned", events)

    def test_duplicate_command_is_suppressed(self) -> None:
        self.make_ready()
        command = self.command()
        self.assertEqual("accepted", self.bridge.submit(command))
        self.assertEqual("duplicate", self.bridge.submit(command))
        self.assertEqual(1, self.bridge.queue_depth)

    def test_queue_overflow_fails_closed(self) -> None:
        self.make_ready()
        self.bridge.submit(self.command("cmd-001"))
        self.bridge.submit(self.command("cmd-002"))
        self.assertEqual("rejected", self.bridge.submit(self.command("cmd-003")))
        self.assertTrue(self.bridge.degraded)
        self.assertEqual(2, self.bridge.queue_depth)

    def test_kill_switch_blocks_admission(self) -> None:
        self.make_ready()
        self.bridge.set_kill_switch(True, reason="operator_test")
        self.assertEqual("rejected", self.bridge.submit(self.command()))
        self.assertEqual("kill_switch_enabled", self.bridge.ledger[-1]["reason"])

    def test_receipt_must_correlate_to_the_command(self) -> None:
        self.make_ready()
        self.bridge.submit(self.command())
        receipt = self.bridge.process_next(
            lambda _: ExecutionReceipt("different-id", "completed"),
        )
        self.assertEqual("failed", receipt.status)
        self.assertEqual("receipt_identity_mismatch", receipt.detail)
        self.assertTrue(self.bridge.degraded)

    def test_reconciliation_records_state_without_execution(self) -> None:
        calls = 0

        def executor(_: Command) -> ExecutionReceipt:
            nonlocal calls
            calls += 1
            return ExecutionReceipt("recovered-001", "completed")

        recovered = self.bridge.reconcile(
            [ExecutionReceipt("recovered-001", "completed", "external_snapshot")],
        )
        self.assertEqual(1, recovered)
        self.assertEqual(0, calls)
        self.assertFalse(self.bridge.ledger[-1]["execution_allowed"])
        self.assertIsNone(self.bridge.process_next(executor))

    def test_reconciliation_ignores_non_terminal_state(self) -> None:
        recovered = self.bridge.reconcile(
            [ExecutionReceipt("recovered-002", "pending", "external_snapshot")],
        )
        self.assertEqual(0, recovered)
        self.assertEqual("non_terminal_status", self.bridge.ledger[-1]["reason"])

    def test_event_ledger_is_valid_jsonl(self) -> None:
        self.make_ready()
        self.bridge.submit(self.command())
        records = [json.loads(line) for line in self.bridge.to_jsonl().splitlines()]
        self.assertEqual(len(self.bridge.ledger), len(records))
        self.assertEqual("command_accepted", records[-1]["event"])
        self.assertIn("ts", records[-1])


if __name__ == "__main__":
    unittest.main()
