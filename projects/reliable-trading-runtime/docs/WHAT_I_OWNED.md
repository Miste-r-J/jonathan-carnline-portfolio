# What I Did

I designed and assembled this system around a practical requirement: I needed to know what the software actually did, not what one status flag claimed it did.

I built the Python runtime that reads market data, creates model inputs, evaluates setups, applies risk and safety rules, creates execution intents, talks to NinjaTrader, reconciles broker state, and writes the audit trail. I also built and maintained the surrounding training, replay, backtest, validation, health, and diagnostic tools.

I worked across Python, C#, PowerShell, Windows, and Linux because the full operating path crossed all of them. I debugged failures by following real artifacts through the system: source bar, feature row, prediction, gate decision, intent, bridge acknowledgement, fill, position snapshot, protection state, and PnL.

## Decisions I made

- I separated prediction from permission to trade.
- I separated connection status from execution readiness.
- I made broker snapshots authoritative after reconnects.
- I used stable IDs and deduplication to make retries safer.
- I blocked new exposure when state could not be proven.
- I kept append-only evidence so failures could be reconstructed.
- I added replay and regression tests for failures that were difficult to reproduce live.
- I kept paper, replay, backfill, and live modes explicit so one mode could not silently behave like another.

## Problems I had to solve

- Data could arrive late, twice, out of order, or with different timestamp formats.
- A socket could remain connected while the platform state was stale.
- An order acknowledgement could arrive without a fill, or a fill could arrive late.
- Local position state could disagree with NinjaTrader after a restart.
- A close request could overlap with protection handling or manual intervention.
- Model features in training could drift from the live feature set.
- Backfill behavior could differ from live behavior if time and freshness rules were not modeled carefully.
- A single summary status could hide the real point of failure.

## What this project demonstrates

This work shows how I approach production-support and reliability problems. I break a large system into observable contracts, identify the source of truth for each state, keep recovery conservative, and build enough evidence to explain failures after they happen. Those habits apply directly to application support, production operations, integration engineering, automation, DevOps, and systems reliability work.

