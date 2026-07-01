# 02 — NinjaTrader Bridge

This was a Windows integration problem as much as a trading problem.

The Python runtime needed to communicate with a C# NinjaTrader add-on, keep order lineage straight, recover across reconnects, and keep enough evidence to explain what happened after the fact.

## What I built

- TCP-style message flow between Python and a C# add-on;
- a HELLO/capability handshake before accepting orders;
- queue handling for inbound, outbound, and pre-handshake messages;
- order acknowledgements with stable client order IDs;
- fill, order-update, position, and PnL snapshot messages;
- execution ledger records for replay/idempotency checks;
- kill-switch behavior when trading is disabled or state is unsafe;
- snapshot freshness checks so "connected" does not mean "trusted."

## Safe public protocol sketch

```json
{
  "type": "ORDER",
  "protocol_version": 1,
  "client_order_id": "stable-id-created-before-send",
  "instrument": "REDACTED",
  "side": "BUY_OR_SELL",
  "qty": 1,
  "intent": "ENTRY_OR_CLOSE"
}
```

```json
{
  "type": "ORDER_ACK",
  "client_order_id": "stable-id-created-before-send",
  "status": "SUBMITTED_OR_REJECTED_OR_DUPLICATE",
  "terminal": false,
  "reason": "machine-readable-reason"
}
```

## Design choices

- Every order needs a stable ID before it leaves Python.
- Duplicate messages should return the prior known result instead of creating a second action.
- If the ledger cannot be written, the system should degrade or lock out.
- Snapshots should be fresh enough to support the decision being made.
- Live routing should be explicit and allowlisted, not accidental.

## What this proves

I can connect a Python runtime to a Windows/C# application, keep the protocol understandable, and build around the messy parts: reconnects, queues, acknowledgements, duplicates, and state mismatch.

