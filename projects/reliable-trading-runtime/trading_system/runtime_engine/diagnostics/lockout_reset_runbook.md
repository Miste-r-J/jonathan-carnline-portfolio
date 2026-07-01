# Sticky Lockout Reset Runbook

1. Stop the current stream process.
2. Reset the lockout with the runtime reset token flow used by your launcher.
3. Confirm in `status.json` that `hard_lockout_active=false` and `hard_lockout_code=null`.
4. Confirm `max_fill_slippage_points` is present and > 0 before re-arming.
5. If authoritative NT PnL is enabled, confirm `pnl_quality_state=ok` before re-arming.
6. During non-flat state, confirm `live_pnl_source=nt_account_api` and advancing `last_pnl_seq`.
7. Restart stream and verify `nt_exec_state=ARMED` before trusting new OPEN emits.
