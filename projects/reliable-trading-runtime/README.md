# Reliable Trading Runtime

I built this system to solve a problem I kept running into: a trading process can look connected and healthy while the market feed, model decision, risk checks, order bridge, broker acknowledgement, fill, and local position state disagree. I wanted one runtime that could make those disagreements visible and stop safely when it could not prove what happened.

This repository contains the actual source I use to train and evaluate models, replay historical sessions, process live market bars, apply entry and account-level controls, send orders through a Python-to-C# NinjaTrader bridge, reconcile execution state, and produce an evidence trail I can audit after a run.

This is independent engineering work. It is not a claim of employment in financial technology, a finished commercial product, or investment advice.

## What I built

- A Python runtime for backfill, replay, paper, and live operating modes.
- A feature and model pipeline with schema checks, label validation, probability calibration, and champion-model loading.
- A deterministic signal-to-order path that records why a candidate was allowed, blocked, changed, or sent.
- A TCP bridge between Python and a C# NinjaTrader 8 AddOn.
- Position, acknowledgement, fill, bracket-protection, and PnL reconciliation.
- Fail-closed checks for stale data, stale broker snapshots, duplicate events, queue degradation, lockouts, and missing protection.
- Append-only JSONL/CSV evidence for signals, gates, execution decisions, orders, fills, positions, and run health.
- Replay, parity, audit, and regression tooling for reproducing failures instead of guessing at them.
- Premarket planning, Discord notifications, and operational diagnostics around the core runtime.

## How the system works

```text
Historical or live bars
        |
        v
Feature calculation and schema validation
        |
        v
Model probability and setup evaluation
        |
        v
Risk, time, freshness, position, and protection gates
        |
        v
Canonical execution intent with a stable correlation ID
        |
        v
Python TCP client <-> C# NinjaTrader AddOn <-> brokerage connection
        |
        v
Acknowledgements, fills, positions, protection state, and PnL
        |
        v
Reconciliation plus append-only audit files and health summaries
```

I deliberately keep prediction, permission, order intent, acknowledgement, and fill as separate states. A model saying `OPEN` does not mean an order was sent. A bridge saying `connected` does not mean an order was accepted. An acknowledgement does not mean a fill occurred. The runtime only advances when the next piece of evidence exists.

## Where to start

| Area | Purpose |
| --- | --- |
| [`simplified/na/discord_addons/cli/stream_live_csv.py`](simplified/na/discord_addons/cli/stream_live_csv.py) | Main runtime and execution state machine |
| [`simplified/na/discord_addons/nt_bridge.py`](simplified/na/discord_addons/nt_bridge.py) | Python-side NinjaTrader protocol and transport |
| [`simplified/na/discord_addons/ninjatrader/NinjaRepoBridge.cs`](simplified/na/discord_addons/ninjatrader/NinjaRepoBridge.cs) | C# AddOn that receives commands and reports broker state |
| [`simplified/na/config/master.yaml`](simplified/na/config/master.yaml) | Central strategy and runtime configuration |
| [`simplified/na/bot/`](simplified/na/bot) | Features, labels, models, training, backtesting, and risk logic |
| [`simplified/na/tests/`](simplified/na/tests) | Regression tests for execution, safety, parity, data, and model contracts |
| [`simplified/tools/`](simplified/tools) | Run audits, parity checks, dataset building, and validation tools |
| [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) | Detailed walkthrough of the runtime |
| [`docs/WHAT_I_OWNED.md`](docs/WHAT_I_OWNED.md) | First-person explanation of my engineering work |
| [`docs/EMPLOYER_BRIEF.md`](docs/EMPLOYER_BRIEF.md) | Recruiter pitch, resume bullets, target roles, and interview walkthrough |

## The engineering problem I focused on

The hardest part was not producing a prediction. It was maintaining a trustworthy state across two languages, multiple processes, a live data stream, a brokerage platform, reconnects, partial failures, and delayed events.

I solved that by using stable correlation IDs, append-only ledgers, atomic file writes, explicit state transitions, bounded retries, broker snapshots, stale-data limits, duplicate suppression, and conservative recovery rules. When local state and broker state conflict, the system blocks new exposure and reconciles before continuing.

## Running the code

The complete live path requires NinjaTrader 8, market data, trained model artifacts, and local configuration that are not distributed in this public repository. The source and test suite are still reviewable without those private runtime dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "$(Resolve-Path .);$(Resolve-Path .\simplified)"
python -m pytest simplified\na\tests -q
```

For a focused review of the execution and reliability work:

```powershell
python -m pytest `
  simplified\na\tests\test_nt_bridge.py `
  simplified\na\tests\test_config_no_duplicate_keys.py `
  simplified\na\tests\test_prop_guardrails_integration.py -q
```

## What I would discuss in an interview

I can walk through a signal from bar ingestion to final fill evidence, explain why I separated desired action from governed action, show how the Python and C# sides recover after a disconnect, and explain how the ledgers identify the exact stage where a run failed. I can also discuss the tradeoffs in the current design, including the size of the main runtime module and how I would split it into smaller services without weakening its safety invariants.

## Public-repository boundaries

I published the implementation, tests, configuration structure, and operational documentation. I excluded credentials, webhook URLs, account identifiers, raw trading logs, private datasets, trained model binaries, generated run folders, and machine-specific secrets. Those items are not necessary to evaluate the engineering and should never be committed to a public repository.
