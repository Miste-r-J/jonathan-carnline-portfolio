# 03 — Audit Bundle Ingestion

The bot created a lot of runtime evidence: status files, JSONL event streams, CSV state, signal-to-order records, execution ledgers, bridge events, and trade records.

That is only useful if it can be pulled into a structure where I can ask better questions.

## What I built

- ZIP/folder ingestion for run artifacts;
- SHA256 hashing of source bundles and source files;
- raw JSONL staging before curated parsing;
- curated tables for signals, gates, signal-to-order decisions, orders, fills, trades, bridge events, and execution events;
- nested ZIP handling for archived event files;
- UTC timestamp normalization with the raw timestamp preserved;
- idempotent inserts and unique keys so re-runs do not duplicate evidence;
- derived orders, fills, and positions from lower-level records.

## Ingestion shape

```text
run export
  -> source file registry
  -> raw record staging
  -> curated tables
  -> derived orders/fills/positions
  -> row counts and investigation queries
```

## Why it mattered

Raw rows can lie if you sum them wrong. Restarts, backfills, replay rows, and overlapping run folders can make the same thing appear more than once.

The audit workflow separated raw records from canonical evidence so I could answer:

- what actually executed;
- what was only planned;
- which rows were duplicates;
- where the execution path broke;
- whether the status file matched the ledger and fill truth.

## What this proves

I can turn messy operational output into a structured investigation path. That matters in production support, application support, incident response, and any operations role where logs are only useful if you can trust the way they were collected.

