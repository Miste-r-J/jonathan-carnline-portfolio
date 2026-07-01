# NT Authoritative PnL Cutover Runbook

## Goal
Verify NinjaTrader AddOn authoritative unrealized PnL feed is healthy before arming live execution.

## Preconditions
1. Confirm lockout is reset.
2. Confirm `status.json` has `hard_lockout_active=false` and `hard_lockout_code=null`.
3. Start streamer with:
   - `--nt_require_pnl_snapshot`
   - `--nt_pnl_snapshot_stale_sec 60`
   - `--nt_pnl_preflight_policy fail`

## Startup checks
1. Confirm `status.json` fields are present:
   - `live_unrealized_pnl`
   - `live_pnl_source`
   - `pnl_feed_staleness_ms`
   - `last_pnl_seq`
   - `pnl_quality_state`
2. Before arming, require:
   - `pnl_quality_state="ok"`
   - `pnl_feed_staleness_ms < 60000`
3. During non-flat position, require:
   - `live_pnl_source="nt_account_api"`
   - `last_pnl_seq` advancing over time

## Paper soak validation
1. Run paper session and collect `nt_bridge.jsonl`.
2. Execute:
```powershell
python -m trading_system.runtime_engine.diagnostics.report_nt_pnl_parity --run-dir <run_dir> --warn-delta-usd 25
```
3. Investigate if:
   - `authoritative_samples == 0`
   - `warn_delta_hits > 0`

## Live go/no-go
Go only if all hold:
1. `hard_lockout_active=false`
2. `pnl_quality_state=ok`
3. `live_pnl_source=nt_account_api` while in position
4. No out-of-order seq warnings in telemetry.

## Rollback
If degraded:
1. Stop run.
2. Set `--nt_pnl_preflight_policy warn` for diagnostics-only restart.
3. Keep execution disarmed until authoritative feed recovers.
