#!/usr/bin/env python3
"""Gap-1 probes (footnote 7): PTX 9.2 u8x4/s8x4 SIMD integer ops.

Doc-miner (results/k8_doc_mining_report.md sec2.4) found:
  "SIMD integer ops add/sub/min/max/neg on .u8x4/.s8x4 and
   add.sat.{u16x2/s16x2/u32} - PTX 9.2, sm_120f only."
This probes every one against ALL SIX targets to confirm/refute "consumer-only"
and pin the exact matrix (is it literally 120f, or the whole 120 family?).
Emits probes_gap.json in runner.py's schema. Operands r0/r1/r2/r3 are .b32
(each holds 4 packed bytes / 2 packed halfwords), declared by STD_DECLS.
"""
import json, os
ALL6 = ["sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_120f", "sm_121a"]
P = []
def add(name, code, ver="9.2"):
    P.append({"family": "G1_simd", "name": name, "code": code,
              "arches": ALL6, "version": ver})

# --- documented PTX 9.2 SIMD byte ops: add/sub/min/max on u8x4/s8x4 ---
for op in ["add", "sub", "min", "max"]:
    for t in ["u8x4", "s8x4"]:
        add(f"{op}.{t}", f"{op}.{t} r0, r1, r2;")
# neg (signed only makes sense; probe both, unsigned expected illegal everywhere)
add("neg.s8x4", "neg.s8x4 r0, r1;")
add("neg.u8x4", "neg.u8x4 r0, r1;")
# saturating add on the wider packed forms (also PTX 9.2 per doc-miner)
add("add.sat.u16x2", "add.sat.u16x2 r0, r1, r2;")
add("add.sat.s16x2", "add.sat.s16x2 r0, r1, r2;")
add("add.sat.u32",   "add.sat.u32 r0, r1, r2;")
add("sub.sat.s8x4",  "sub.sat.s8x4 r0, r1, r2;")
add("add.sat.s8x4",  "add.sat.s8x4 r0, r1, r2;")
add("add.sat.u8x4",  "add.sat.u8x4 r0, r1, r2;")

# --- absdiff: task named "vabsdiff"; probe modern packed + legacy video spellings ---
add("vabsdiff.u8x4", "vabsdiff.u8x4 r0, r1, r2;")
add("vabsdiff.s8x4", "vabsdiff.s8x4 r0, r1, r2;")
add("absdiff.u8x4",  "absdiff.u8x4 r0, r1, r2;")
add("absdiff.s8x4",  "absdiff.s8x4 r0, r1, r2;")
add("vabsdiff4.u32.u32.u32",      "vabsdiff4.u32.u32.u32 r0, r1, r2, r3;")
add("vabsdiff4.u32.u32.u32.add",  "vabsdiff4.u32.u32.u32.add r0, r1, r2, r3;")
add("vabsdiff2.u32.u32.u32",      "vabsdiff2.u32.u32.u32 r0, r1, r2, r3;")

# --- positive controls (validate the harness/template; must ACCEPT broadly) ---
add("CTRL.add.s32",    "add.s32 r0, r1, r2;")                          # baseline all archs
add("CTRL.add.s16x2",  "add.s16x2 r0, r1, r2;")                        # packed 16b (pre-9.2)
add("CTRL.min.s16x2",  "min.s16x2 r0, r1, r2;")
add("CTRL.max.s16x2",  "max.s16x2 r0, r1, r2;")
add("CTRL.vadd4.u32.u32.u32.add", "vadd4.u32.u32.u32.add r0, r1, r2, r3;")  # legacy video SIMD
add("CTRL.dp4a.u32.u32", "dp4a.u32.u32 r0, r1, r2, r3;")               # byte dot (widely present)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes_gap.json")
json.dump(P, open(out, "w"), indent=1)
print(len(P), "probes ->", out)
