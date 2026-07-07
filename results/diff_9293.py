#!/usr/bin/env python3
"""Diff the PTX 9.2 (ptxas 13.2) verdict matrix against the PTX 9.3 (ptxas 13.3)
re-run of the SAME 993 probes. Surfaces the toolchain-bump findings:

  * FLIP  -- a cell that was REJECT under 13.2 and ACCEPT under 13.3 (or the
             reverse): the "newer toolchains differ again" case the 9.2 matrix flagged.
  * CLASS -- a reject whose class changed (e.g. UNKNOWN-INSTR -> ARCH), which
             can mean an instruction the old ptxas didn't know is now a real
             arch answer.
  * ONLY  -- probes present in only one matrix (coverage delta).

Usage: python results/diff_9293.py [old_9.2.json] [new_9.3.json]
Defaults: results/results_full.json  vs  results/results_full_ptx93.json
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
old_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_full.json")
new_p = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results_full_ptx93.json")
old = json.load(open(old_p))
new = json.load(open(new_p))
ARCHES = ["sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_120f", "sm_121a"]

def cell(res, a):
    r = res.get(a, {})
    return "ACCEPT" if r.get("status") == "ACCEPT" else "REJECT:" + r.get("class", "?")

flips, classchg = [], []
for k in sorted(set(old) & set(new)):
    o, n = old[k]["res"], new[k]["res"]
    for a in ARCHES:
        co, cn = cell(o, a), cell(n, a)
        if co == cn:
            continue
        oa, na = co.startswith("ACCEPT"), cn.startswith("ACCEPT")
        (flips if oa != na else classchg).append((k, a, co, cn))

only_old = sorted(set(old) - set(new))
only_new = sorted(set(new) - set(old))

print(f"9.2 matrix: {old_p}  ({len(old)} probes)")
print(f"9.3 matrix: {new_p}  ({len(new)} probes)")
print(f"\n=== ACCEPT<->REJECT FLIPS (the headline) : {len(flips)} ===")
for k, a, co, cn in flips:
    print(f"  {k:55} {a:9} {co:16} -> {cn}")
print(f"\n=== reject-CLASS changes : {len(classchg)} ===")
for k, a, co, cn in classchg:
    print(f"  {k:55} {a:9} {co:16} -> {cn}")
print(f"\n=== probes only in 9.2 matrix: {len(only_old)} ; only in 9.3 matrix: {len(only_new)} ===")
if not flips and not classchg:
    print("\nNo per-cell verdict changed 9.2 -> 9.3 across the 993-probe common set.")
    print("(Interpretation: 13.3 did not move any documented-9.2 instruction's")
    print(" six-target verdict; all 9.3 news is additive -- see results_ptx93.json.)")
