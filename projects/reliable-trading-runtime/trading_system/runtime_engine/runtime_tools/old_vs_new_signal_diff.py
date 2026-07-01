from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _canon_ts(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").isoformat()
    except Exception:
        return str(value)


def _manifest_meta(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "run_manifest.json")
    resolved = _read_json(run_dir / "resolved_config.json")
    argv = manifest.get("cli_argv") or []
    phase2_tag = resolved.get("phase2_tag")
    if "--phase2_tag" in argv:
        i = argv.index("--phase2_tag")
        if i + 1 < len(argv):
            phase2_tag = argv[i + 1]
    return {
        "preset": manifest.get("preset") or resolved.get("preset") or "",
        "phase2_tag": phase2_tag or "",
        "threshold_sources": json.dumps(manifest.get("threshold_sources") or resolved.get("threshold_sources") or {}),
    }


def _build_index(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_ts: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(run_dir / "signal_to_order.jsonl"):
        ts = _canon_ts(rec.get("bar_ts") or rec.get("datetime") or rec.get("ts"))
        if not ts:
            continue
        ctx = rec.get("ctx") if isinstance(rec.get("ctx"), dict) else {}
        phase2 = ctx.get("phase2") if isinstance(ctx.get("phase2"), dict) else {}
        risk = rec.get("risk") if isinstance(rec.get("risk"), dict) else {}
        by_ts.setdefault(ts, {}).update(
            {
                "price": rec.get("price"),
                "action": str(rec.get("action") or rec.get("type") or "").upper(),
                "side": str(rec.get("side") or "").upper(),
                "final_action": str(rec.get("final_action") or rec.get("type") or "").upper(),
                "setup_pass": ctx.get("phase2_setup_pass", rec.get("phase2_setup_pass")),
                "setup_prob": phase2.get("setup_prob", rec.get("setup_prob")),
                "direction_prob": rec.get("directional_prob", rec.get("prob")),
                "blocked_by": ";".join(rec.get("_signal_blocked_by") or []),
                "reason_detail": rec.get("reason_detail") or "",
                "legacy_bypass": ctx.get("phase2_force_open_legacy_gate_bypass", rec.get("phase2_force_open_legacy_gate_bypass")),
                "aggressive_allowed": bool(ctx.get("aggressive_directional_bridge")),
                "stop": risk.get("stop"),
                "target": risk.get("target"),
                "open_close_flip": str(rec.get("type") or "").upper(),
            }
        )
    for rec in _read_jsonl(run_dir / "gating_events.jsonl"):
        ts = _canon_ts(rec.get("bar_ts") or rec.get("datetime") or rec.get("ts"))
        if not ts:
            continue
        cur = by_ts.setdefault(ts, {})
        if not cur.get("blocked_by"):
            cur["blocked_by"] = str(rec.get("blocked_by") or "")
        if not cur.get("reason_detail"):
            cur["reason_detail"] = str(rec.get("reason_detail") or "")
    return by_ts, _manifest_meta(run_dir)


def _classify(old: dict[str, Any], new: dict[str, Any]) -> str:
    oa = str(old.get("action") or "").upper()
    na = str(new.get("action") or "").upper()
    os = str(old.get("side") or "").upper()
    ns = str(new.get("side") or "").upper()
    if not old or not new:
        return "DATA_DIFFERENT"
    if oa == na and os == ns and str(old.get("stop")) == str(new.get("stop")) and str(old.get("target")) == str(new.get("target")):
        return "SAME"
    if oa == "OPEN" and na in {"", "HOLD", "NO_TRADE"}:
        return "OLD_OPEN_NEW_NO_TRADE"
    if oa == "FLIP" and na in {"", "HOLD", "NO_TRADE"}:
        return "OLD_FLIP_NEW_HOLD"
    if os == "LONG" and ns == "SHORT":
        return "OLD_LONG_NEW_SHORT"
    if os == "SHORT" and ns == "LONG":
        return "OLD_SHORT_NEW_LONG"
    if oa in {"CLOSE", "EXIT"} and na in {"", "HOLD", "NO_TRADE"}:
        return "OLD_CLOSE_NEW_HOLD"
    if oa in {"", "HOLD", "NO_TRADE"} and na == "OPEN":
        return "OLD_HOLD_NEW_OPEN"
    if str(old.get("stop")) != str(new.get("stop")) or str(old.get("target")) != str(new.get("target")):
        return "STOP_TARGET_DIFFERENT"
    if str(old.get("setup_pass")) != str(new.get("setup_pass")):
        return "SETUP_GATE_DIFFERENT"
    if str(old.get("direction_prob")) != str(new.get("direction_prob")):
        return "DIRECTION_GATE_DIFFERENT"
    if str(old.get("blocked_by")) != str(new.get("blocked_by")):
        return "FLIP_RULE_DIFFERENT" if "flip" in (str(old.get("reason_detail")) + str(new.get("reason_detail"))).lower() else "DATA_DIFFERENT"
    return "DATA_DIFFERENT"


def build_diff(old_run: Path, new_run: Path, out_csv: Path) -> None:
    old_index, old_meta = _build_index(old_run)
    new_index, new_meta = _build_index(new_run)
    timestamps = sorted(set(old_index) | set(new_index))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp", "price", "old_action", "new_action", "old_side", "new_side", "old_final_action", "new_final_action",
        "old_setup_pass", "new_setup_pass", "old_setup_prob", "new_setup_prob", "old_direction_probability", "new_direction_probability",
        "old_blocked_by", "new_blocked_by", "old_reason_detail", "new_reason_detail", "old_phase2_force_open_legacy_gate_bypass",
        "new_aggressive_bridge_allowed", "old_stop", "new_stop", "old_target", "new_target", "old_mode_preset", "new_mode_preset",
        "old_open_close_flip", "new_open_close_flip", "old_phase2_tag", "new_phase2_tag", "old_threshold_sources", "new_threshold_sources",
        "diff_classification",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ts in timestamps:
            old = old_index.get(ts, {})
            new = new_index.get(ts, {})
            writer.writerow(
                {
                    "timestamp": ts,
                    "price": old.get("price", new.get("price")),
                    "old_action": old.get("action", ""),
                    "new_action": new.get("action", ""),
                    "old_side": old.get("side", ""),
                    "new_side": new.get("side", ""),
                    "old_final_action": old.get("final_action", ""),
                    "new_final_action": new.get("final_action", ""),
                    "old_setup_pass": old.get("setup_pass", ""),
                    "new_setup_pass": new.get("setup_pass", ""),
                    "old_setup_prob": old.get("setup_prob", ""),
                    "new_setup_prob": new.get("setup_prob", ""),
                    "old_direction_probability": old.get("direction_prob", ""),
                    "new_direction_probability": new.get("direction_prob", ""),
                    "old_blocked_by": old.get("blocked_by", ""),
                    "new_blocked_by": new.get("blocked_by", ""),
                    "old_reason_detail": old.get("reason_detail", ""),
                    "new_reason_detail": new.get("reason_detail", ""),
                    "old_phase2_force_open_legacy_gate_bypass": old.get("legacy_bypass", ""),
                    "new_aggressive_bridge_allowed": new.get("aggressive_allowed", ""),
                    "old_stop": old.get("stop", ""),
                    "new_stop": new.get("stop", ""),
                    "old_target": old.get("target", ""),
                    "new_target": new.get("target", ""),
                    "old_mode_preset": old_meta.get("preset", ""),
                    "new_mode_preset": new_meta.get("preset", ""),
                    "old_open_close_flip": old.get("open_close_flip", ""),
                    "new_open_close_flip": new.get("open_close_flip", ""),
                    "old_phase2_tag": old_meta.get("phase2_tag", ""),
                    "new_phase2_tag": new_meta.get("phase2_tag", ""),
                    "old_threshold_sources": old_meta.get("threshold_sources", ""),
                    "new_threshold_sources": new_meta.get("threshold_sources", ""),
                    "diff_classification": _classify(old, new),
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Build old vs new signal diff CSV")
    p.add_argument("old_run", type=Path)
    p.add_argument("new_run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build_diff(args.old_run, args.new_run, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
