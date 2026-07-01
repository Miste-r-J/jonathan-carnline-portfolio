# Phase-2 Live Readiness Checklist

Use this checklist before promoting `es_elite_v1_live_ready` beyond paper mode.

## Bundle
- Candidate tag exists: `retrain_v6_fixed_v2_hyper_016_livebundle_v1`
- Candidate directory contains `setup.joblib`, `dir.joblib`, `close.joblib`, and the close metadata files
- `manifest.json` uses candidate-local paths for `setup_model_path`, `dir_model_path`, and `close_model_path`
- `promotion_result` is non-pending

## Parity
- `python simplified/tools/check_deployable_phase2_candidate.py --tag retrain_v6_fixed_v2_hyper_016_livebundle_v1`
- `python simplified/tools/phase2_rebuild_parity_audit.py --preset es_elite_v1_live_ready --resolved-config <resolved_config.json>`
- Confirm the audit reports:
  - `phase2_manifest_thresholds_used=true`
  - all threshold sources = `phase2_tag`
  - `phase2_close.enabled=true`
  - close model path inside the active candidate directory
  - explicit TOD window instead of `gate_tod_disabled_auto_24h`

## Launcher
- Use `--preset es_elite_v1_live_ready`
- Start with `--nt_exec_policy paper`
- Require NT connection and snapshot before evaluation
- Keep `--live_only_stale_gate`
- Do not cut over if startup logs show the close model disabled or downgraded

## Rollout
- Paper soak: 2 full sessions minimum with no threshold-source, close-load, or stale-gate regression
- Shadow review: compare close-model actions and execution artifacts before live cutover
- Live cutover only after paper and shadow are clean
