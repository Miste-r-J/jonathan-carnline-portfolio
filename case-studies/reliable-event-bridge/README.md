# Reliable Event Bridge

An original, cleaned-up reference implementation of the reliability patterns behind a private Python/C# integration.

The project demonstrates how to keep a cross-system command path observable and fail closed when connections reset, requests replay, queues fill, receipts cannot be correlated, or external state must be recovered.

> This is a portfolio case study, not production source. It contains no strategy logic, private research, credentials, account data, production configuration, or private event records.

## Why this exists

A system can report that it is connected while still being unsafe or unable to prove that a command completed. The engineering problem is to separate:

1. transport connectivity;
2. handshake readiness;
3. command admission;
4. downstream execution;
5. terminal acknowledgement; and
6. recovered external state.

This repository makes those boundaries explicit and testable.

## Reliability behavior demonstrated

- **Fail-closed handshake:** commands are rejected until the current generation and session nonce are acknowledged.
- **Generation-aware reconnects:** work from an older connection cannot enter the active queue.
- **Idempotent admission:** durable command IDs suppress retry and replay duplicates.
- **Bounded backpressure:** queue overflow creates an explicit rejection and degraded state.
- **Strict correlation:** a receipt cannot complete a different command.
- **Reconcile-only recovery:** externally recovered terminal state is recorded without repeating the action.
- **Operational kill switch:** operators can stop new admission without ambiguous state.
- **Evidence-led observability:** every lifecycle decision is recorded as structured JSONL.

See [the architecture](docs/architecture.md) for the contracts and failure modes.

## Run locally

Requirements: Python 3.11 or newer.

```bash
python -m pip install .
python -m unittest discover -s tests -v
python -m reliable_event_bridge.demo
```

The test suite uses only Python's standard library. The GitHub Actions workflow runs the same verification on Python 3.11, 3.12, and 3.13.

## Repository map

```text
.
├── .github/workflows/ci.yml     # Repeatable verification
├── docs/architecture.md         # System and sequence diagrams
├── src/reliable_event_bridge/
│   ├── bridge.py                # Reliability contracts
│   └── demo.py                  # Synthetic demonstration
├── tests/test_bridge.py         # Contract-focused tests
├── EVIDENCE.md                  # Proof and disclosure boundary
├── INTERVIEW_NOTES.md           # Concise technical walkthrough
├── PUBLICATION_CHECKLIST.md      # Safe GitHub handoff
└── SECURITY.md                  # Pre-publication safety checklist
```

## Example evidence trail

```json
{"event":"connection_started","generation":1}
{"event":"handshake_accepted","generation":1}
{"action":"APPLY_CONFIGURATION","command_id":"demo-001","event":"command_accepted","generation":1,"queue_depth":1}
{"command_id":"demo-001","event":"duplicate_suppressed","generation":1}
{"command_id":"demo-001","detail":"adapter_ack","event":"execution_terminal","status":"completed"}
```

Runtime records also contain UTC timestamps. They are omitted above only to keep the lifecycle easy to scan.

## Design choices

### “Connected” is not “ready”

The bridge requires an acknowledgement tied to the current connection identity before accepting work. This prevents an old or partial session from being treated as healthy.

### Replays are expected

Retries and reconnects can replay requests. The command ID is the durable identity; duplicate admission returns a stable disposition without another downstream side effect.

### Unknown state does not authorize execution

Recovered receipts update the evidence trail through reconciliation. They never enter the execution queue.

### Backpressure is visible

When capacity is exhausted, the bridge rejects new work and marks itself degraded. It does not silently drop or accept unbounded work.

## Publication boundary

The private project was reviewed only to identify the reliability patterns I could safely explain in public. The public code was written separately for this case study. See [EVIDENCE.md](EVIDENCE.md) and [SECURITY.md](SECURITY.md) for the disclosure boundary.

## Author

Jonathan Carnline — operations, automation, integration, and production reliability.
