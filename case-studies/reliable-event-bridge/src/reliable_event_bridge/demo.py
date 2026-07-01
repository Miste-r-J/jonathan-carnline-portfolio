"""Run a short reliability demonstration from the command line."""

from .bridge import Command, ExecutionReceipt, ReliableEventBridge


def main() -> None:
    bridge = ReliableEventBridge(max_queue=2)
    generation = bridge.connect(session_nonce="demo-session")
    bridge.acknowledge_handshake(
        generation=generation,
        session_nonce="demo-session",
    )

    command = Command(
        command_id="demo-001",
        generation=generation,
        action="APPLY_CONFIGURATION",
        payload={"component": "example-service", "mode": "safe"},
    )
    bridge.submit(command)
    bridge.submit(command)  # deliberately replayed; suppressed by command ID
    bridge.process_next(
        lambda item: ExecutionReceipt(item.command_id, "completed", "adapter_ack"),
    )

    print(bridge.to_jsonl())


if __name__ == "__main__":
    main()

