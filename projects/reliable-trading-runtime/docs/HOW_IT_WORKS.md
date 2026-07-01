# How I Built the Runtime

## 1. Market data enters through a controlled source

I support CSV tailing, replay data, backfill data, and a live socket feed. Before a bar can affect a decision, I normalize its timestamp, validate its OHLCV fields, reject malformed or duplicate rows, and track freshness. This keeps the downstream pipeline from silently operating on bad or stale input.

## 2. I calculate features under a fixed contract

The feature code builds price, volume, volatility, session, multi-timeframe, order-flow, and setup context. The model bundle carries the expected feature list and schema information. At runtime I align columns, reject incompatible schemas, and record a feature hash so the live path can be compared with replay and training.

## 3. The model proposes; the runtime governs

The prediction layer produces probabilities and a proposed action. That proposal then passes through setup requirements, time-of-day rules, market alignment, freshness checks, cooldowns, position state, account limits, and protection state.

I store both the candidate action and the final governed action. That distinction matters during diagnosis: I can tell whether the model did nothing, the strategy rejected a setup, a safety rule blocked it, or execution failed later.

## 4. Every execution intent receives an identity

Before sending an order, I create a canonical intent containing the instrument, side, quantity, timestamps, operating mode, reason, and a stable correlation ID. That ID follows the order through the Python client, C# bridge, acknowledgement, fill, position update, protection orders, and ledger records.

The identity makes retries safer. If a message is repeated after a timeout or reconnect, the bridge can recognize an already-seen intent instead of creating a second order.

## 5. Python and NinjaTrader communicate over a defined protocol

The Python runtime owns strategy evaluation and safety governance. The C# AddOn owns the platform connection and translates canonical commands into NinjaTrader operations. It returns structured messages for readiness, acknowledgements, order state, fills, positions, account state, and protection status.

I treat transport health and trading readiness as different conditions. A socket can be connected while the account snapshot is stale or the platform is not ready to accept an order.

## 6. Broker state is authoritative for execution

Local state is useful, but it cannot be the final authority after a disconnect or restart. The runtime compares local expectations with NinjaTrader snapshots. If the states disagree, it prevents new entries, repairs what can be repaired safely, and records the conflict for review.

That reconciliation covers positions, working orders, fills, bracket protection, and PnL. Manual flattening and late fills are handled as explicit lifecycle events rather than hidden corrections.

## 7. Failures become evidence, not vague status messages

The runtime writes separate evidence streams for model state, gate decisions, execution intents, bridge messages, order events, fills, positions, and health. Files are append-only where possible, and summary files are written atomically.

When a run does not trade, I can trace:

1. whether a candidate existed;
2. which gate produced the final action;
3. whether an intent was emitted;
4. whether NinjaTrader acknowledged it;
5. whether a fill occurred;
6. whether position and protection snapshots agreed; and
7. whether the final PnL came from broker-authoritative data.

## 8. Replay and tests protect live behavior

I use historical replay and targeted regression tests to reproduce specific failures. The tests cover stale snapshots, duplicate events, bracket geometry, lockouts, reconnects, queue degradation, late fills, manual flattening, feature parity, label schemas, training contracts, and backfill-to-live transitions.

The goal is not only test coverage. The goal is to preserve the exact safety behavior that matters when timing and partial failure make the live path difficult to reason about.

