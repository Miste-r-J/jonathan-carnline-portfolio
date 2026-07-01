# 01 — Execution Safety

The important part of this work was not "make it trade." The important part was making sure it knew when not to act.

The private system had to deal with live state, delayed snapshots, duplicate events, fills that could arrive out of order, and state that could look fine while the actual execution path was not safe.

## What I built

- hard disarm behavior when execution state was uncertain;
- fill and order reconciliation checks;
- status files that explain why the system is armed, blocked, degraded, or stopped;
- protection checks for open positions;
- lockout handling when the bridge or ledger state cannot be trusted;
- regression tests around stale snapshots, stale fills, late protection updates, and reconciliation.

## The pattern

```text
Signal is not enough.
Connected is not enough.
Ready is not enough.

The system needs evidence:
  intent -> gate decision -> order acknowledgement -> fill -> position snapshot -> ledger state

If those do not line up, the safe answer is no new order.
```

## Failure modes I accounted for

- a stale broker snapshot says one thing while the local ledger says another;
- an entry fill arrives before protection order evidence is stable;
- a duplicate or replayed order tries to run again after restart;
- a CLOSE or FLATTEN path gets suppressed by ordinary dedupe logic;
- a live data feed stops producing real bar progress while still looking connected;
- a position is flat but working orders are still present;
- status reporting says "healthy" but execution evidence does not support it.

## What this proves

I can work on systems where the cost of a wrong assumption matters. I do not treat a green status light as proof. I trace the actual state change and make the system tell me why it is safe or why it is blocked.

