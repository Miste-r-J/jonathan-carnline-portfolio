import argparse, json, os, sys, re
from collections import defaultdict

def read_jsonl(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for i,line in enumerate(f, start=1):
            line=line.strip()
            if not line: 
                continue
            try:
                rows.append((i, json.loads(line)))
            except Exception as e:
                raise RuntimeError(f"Failed parsing {path}:{i}: {e}")
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory containing order_events.jsonl etc.")
    ap.add_argument("--protection_timeout_sec", type=float, default=None,
                    help="Optional: expected protection timeout; used for extra warnings only.")
    args=ap.parse_args()

    run_dir=args.run_dir
    order_events_path=os.path.join(run_dir, "order_events.jsonl")
    if not os.path.exists(order_events_path):
        print(f"ERROR: missing {order_events_path}")
        return 1

    events=read_jsonl(order_events_path)

    # Invariants:
    # 1) No flatten_due_to_no_protection
    flatten_no_prot=[(ln,e) for ln,e in events if str(e.get("event","")).lower()=="flatten_due_to_no_protection"]
    # 2) No lockout set to nt_protection_timeout
    lockouts=[(ln,e) for ln,e in events if "lockout" in str(e.get("event","")).lower() or "hard_lockout" in str(e.get("event","")).lower()]
    nt_lockouts=[(ln,e) for ln,e in lockouts if "nt_protection_timeout" in json.dumps(e).lower()]

    # 3) If there is an entry fill, we should see protection confirmed (unless the trade closed instantly).
    entry_fills=[(ln,e) for ln,e in events if str(e.get("event","")).lower() in ("entry_filled","fill","order_filled")]
    prot_conf=[(ln,e) for ln,e in events if "protection confirmed" in str(e.get("event","")).lower() or str(e.get("event","")).lower() in ("protection_confirmed","exits_working_inferred","exits_working")]
    # 4) Duplicate entry spam: count tx bracket/open events grouped by signal_id (if present)
    tx=[(ln,e) for ln,e in events if str(e.get("event","")).lower() in ("tx_order","tx_bracket","submit_order","nt_tx_order","nt_tx_bracket","order_sent")]
    by_signal=defaultdict(list)
    for ln,e in tx:
        sid=e.get("signal_id") or e.get("sid") or None
        cid=e.get("client_order_id") or e.get("cid") or None
        key=sid if sid is not None else cid
        by_signal[key].append((ln,e))

    spam=[(k,v) for k,v in by_signal.items() if k is not None and len(v)>1]

    failed=False

    if flatten_no_prot:
        failed=True
        print("FAIL: flatten_due_to_no_protection occurred:")
        for ln,e in flatten_no_prot[:5]:
            print(f"  line {ln}: {e}")

    if nt_lockouts:
        failed=True
        print("FAIL: nt_protection_timeout lockout evidence found:")
        for ln,e in nt_lockouts[:5]:
            print(f"  line {ln}: {e}")

    if entry_fills and not prot_conf:
        # Not always fatal (e.g., if close happens immediately), but should be investigated.
        print("WARN: entry fill(s) present but no protection confirmation event found. Investigate logs.")
        for ln,e in entry_fills[:3]:
            print(f"  entry fill line {ln}: {e}")

    if spam:
        # This is a strong smell, but could be intentional if different signal_ids are missing.
        print("WARN: multiple TX events for the same signal_id/client_order_id key:")
        for k,v in spam[:5]:
            print(f"  key={k} count={len(v)} first_lines={[x[0] for x in v[:3]]}")
        # mark fail only if signal_id exists (real dedupe should prevent it)
        if any(k is not None and k!="None" and "signal" in str(k).lower() for k,_ in spam):
            failed=True

    if not failed:
        print("PASS: No false protection-flatten or nt_protection_timeout lockout detected.")
        if prot_conf:
            ln,e=prot_conf[0]
            print(f"  Found protection confirmation event at line {ln}: {e.get('event')}")
        if spam:
            print("  Note: there were TX duplicates by key, but treated as warnings (missing/unstable ids).")

    return 2 if failed else 0

if __name__=="__main__":
    sys.exit(main())
