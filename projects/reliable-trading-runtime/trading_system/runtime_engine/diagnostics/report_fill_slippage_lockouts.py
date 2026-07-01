#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _extract_open_fill_slippage_points(nt_bridge_path: Path) -> Dict[str, float]:
    model_price_by_cid: Dict[str, float] = {}
    slippage_by_cid: Dict[str, float] = {}

    for row in _read_jsonl(nt_bridge_path):
        msg = row.get('msg') if isinstance(row, dict) else None
        if not isinstance(msg, dict):
            continue
        mtype = str(msg.get('type') or '').upper()
        cid = str(msg.get('client_order_id') or '')
        if not cid:
            continue

        if mtype == 'ORDER':
            action = str(msg.get('action') or '').upper()
            if action in {'BUY', 'SELL'}:
                mp = msg.get('model_price')
                try:
                    model_price_by_cid[cid] = float(mp)
                except Exception:
                    pass
        elif mtype == 'FILL':
            side = str(msg.get('side') or '').upper()
            if side in {'LONG', 'SHORT'} and cid in model_price_by_cid:
                try:
                    fill_price = float(msg.get('fill_price'))
                    slippage_by_cid[cid] = abs(fill_price - model_price_by_cid[cid])
                except Exception:
                    pass
    return slippage_by_cid


def main() -> int:
    ap = argparse.ArgumentParser(description='Report would-lockout counts for fill slippage thresholds.')
    ap.add_argument('--run-dir', required=True, help='Run directory containing nt_bridge.jsonl')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    nt_bridge = run_dir / 'nt_bridge.jsonl'
    slips = _extract_open_fill_slippage_points(nt_bridge)

    if not slips:
        print(json.dumps({'run_dir': str(run_dir), 'error': 'no_open_fills_found'}, indent=2))
        return 1

    vals = list(slips.values())
    report = {
        'run_dir': str(run_dir),
        'open_fill_count': len(vals),
        'max_slippage_points': max(vals),
        'avg_slippage_points': sum(vals) / len(vals),
        'would_lockout': {
            'threshold_2_0': sum(1 for v in vals if v > 2.0),
            'threshold_6_0': sum(1 for v in vals if v > 6.0),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
