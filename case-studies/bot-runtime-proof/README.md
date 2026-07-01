# Bot Runtime Proof — Public Portfolio Notes

This folder is the public-safe version of selected work from my private bot repo.

I did not publish the production trading system. I pulled out the parts that show the engineering work a hiring team should care about:

- how I think about safe execution;
- how I track state, fills, orders, and failures;
- how I move messy runtime output into something reviewable;
- how I keep long-running work observable instead of guessing from a green light;
- how I document incidents and recovery.

## What I selected

| Area | What it proves | Public folder |
| --- | --- | --- |
| Execution safety | I design systems to fail closed when state is uncertain. | `01-execution-safety/` |
| NinjaTrader bridge | I can integrate Python, C#, Windows apps, and line-delimited event protocols. | `02-ninjatrader-bridge/` |
| Audit bundle ingestion | I can turn run folders and exported logs into queryable evidence. | `03-audit-bundle-ingestion/` |
| Fiber/live feed observability | I can debug live data flow, backpressure, stale feeds, and recovery. | `04-feed-observability/` |
| Discord operations taskboard | I can keep long-running operational work organized across handoffs. | `05-discord-operations-taskboard/` |

## What I left private

- production strategy logic;
- model thresholds and trading research;
- account identifiers;
- credentials, tokens, webhook URLs, and environment files;
- raw fills, live logs, screenshots, and private config;
- anything that would make the system copyable instead of reviewable.

The point here is simple: show the engineering judgment without handing out the private playbook.

