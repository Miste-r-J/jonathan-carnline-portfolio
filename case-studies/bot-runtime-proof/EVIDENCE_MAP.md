# Evidence Map

These notes came from a deep audit of `C:\Users\miste\Documents\bot`.

I treated the bot repo as sensitive and selected only portfolio-safe proof. The original repo includes live run folders, logs, CSV/JSONL traces, screenshots, configs, archives, and private runtime state. Those stayed out.

## Selected source areas

| Private repo area reviewed | Portfolio-safe takeaway |
| --- | --- |
| `simplified/na/discord_addons/cli/stream_live_csv.py` | Runtime gating, stale-feed handling, status writing, reconciliation, and live safety checks. |
| `simplified/na/tests/test_stream_live_csv_regressions.py` | Regression coverage around stale bars, snapshots, fiber backpressure, lockouts, status fields, and reconciliation. |
| `NinjaRepoBridge.cs` | C# add-on bridge patterns: HELLO/capability handshake, kill switch, queues, snapshots, order/fill lineage, and ledger-based idempotency. |
| `ingest_zip/` | Audit-bundle ingestion into structured tables with source hashing, raw staging, curated loads, nested ZIP handling, and derived orders/fills/positions. |
| `docs/EXECUTION_INVARIANTS.md` and `docs/INCIDENT_RESPONSE.md` | Safety rules and operator response patterns. |
| `trade_audit_june2026/supporting_data/DATA_GUIDE.md` | Evidence-first analysis workflow and canonical-vs-raw data handling. |

## Publishing decision

I published explanations, architecture notes, review checklists, and safe pseudocode-level snippets. I did not publish production code that exposes a working trading edge, live execution details, private IDs, or raw operating data.

