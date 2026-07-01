# Employer Brief

## My short explanation

I built a Python and C# runtime that takes market data through feature calculation, model evaluation, safety checks, order execution, broker reconciliation, and an auditable record of what happened. The difficult part was keeping state trustworthy across live data, two languages, reconnects, delayed events, and partial failures. I built stable order identities, fail-closed controls, broker-authoritative reconciliation, structured ledgers, replay tools, and regression tests so I could trace a failure to the exact stage instead of guessing.

## My 30-second version

I work at the intersection of software and operations. My main project is a real-time Python system connected to NinjaTrader through C#. I built the data pipeline, model runtime, safety gates, execution bridge, reconciliation, monitoring, and test tooling. It taught me how to debug stateful production systems where a connected status is not enough—you need evidence that the right event moved through every stage.

## Resume bullets I can use

- Built a Python/C# real-time runtime spanning market-data ingestion, feature engineering, model evaluation, risk governance, NinjaTrader execution, reconciliation, and structured operational telemetry.
- Designed a deterministic signal-to-order workflow with stable correlation IDs, idempotent retries, duplicate suppression, stale-state controls, broker-authoritative snapshots, and fail-closed recovery behavior.
- Created replay, backfill, parity, audit, and regression tooling across 256 Python files and 76 test modules to reproduce failures and validate execution, data, configuration, and model contracts.
- Integrated Python services with a C# NinjaTrader AddOn and separated connection, readiness, acknowledgement, fill, position, protection, and PnL into independently observable states.
- Automated run-health reporting and append-only evidence across signals, gates, execution decisions, orders, fills, positions, and PnL to support root-cause analysis.

## Roles where this work is directly relevant

- Application Support Engineer
- Production Support Engineer
- Systems Operations Engineer
- Reliability Engineer
- Integration Engineer
- Automation Engineer
- Technical Operations Analyst
- Python Developer
- Junior DevOps Engineer
- Trading Systems Support
- Logistics or Warehouse Systems Analyst

## Interview walkthrough

1. I start with the operational problem: surface-level health did not prove execution.
2. I draw the end-to-end event path from market bar to broker fill.
3. I explain the source of truth at each stage.
4. I show why correlation IDs and append-only records matter after retries or reconnects.
5. I explain one failure mode, such as a stale broker snapshot or late fill, and how the runtime blocks and reconciles.
6. I open the tests that preserve that behavior.
7. I close with the design tradeoff: the main runtime is large because the system evolved around live incidents, and my next refactor would extract protocol, reconciliation, protection, and evidence-writing components behind stable interfaces.

## Claims I should not make

- I should not describe this independent project as paid fintech employment.
- I should not promise profitability or present backtest results as guaranteed live results.
- I should not claim every historical test is a continuously supported product test.
- I should not say the repository includes production credentials, datasets, trained models, or a one-command live deployment.

