#!/usr/bin/env python3
"""Verify a convert-path claim from a consumer block-scaled-fp4 attention kernel's
design: 'there is no direct e2m1->e4m3 convert; an fp8-compute variant of the
recipe costs more instructions because it must round-trip through f16/f32.'
Probe direct fp4<->fp8 and fp6<->fp8 packed converts on all six targets.
Expected: no such format-to-format convert exists (values round-trip via
f16/f32) -- the reason a datacenter fp8-scaling recipe does not map one-to-one
onto the consumer block-scaled fp4 path.
e2m1x2 lives in .b8 (q), e4m3x2/e5m2x2/e3m2x2/e2m3x2 in .b16 (h).
"""
import json, os
ALL6 = ["sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_120f", "sm_121a"]
P = []
def add(name, code, ver="9.2"):
    P.append({"family": "G3_conv", "name": name, "code": code,
              "arches": ALL6, "version": ver})

# fp4 -> fp8 (the convert NVIDIA's design would need on consumer)
add("e4m3x2.e2m1x2", "cvt.rn.satfinite.e4m3x2.e2m1x2 h0, q0;")
add("e5m2x2.e2m1x2", "cvt.rn.satfinite.e5m2x2.e2m1x2 h0, q0;")
add("e4m3.e2m1",     "cvt.rn.satfinite.e4m3.e2m1 h0, q0;")
# fp8 -> fp4
add("e2m1x2.e4m3x2", "cvt.rn.satfinite.e2m1x2.e4m3x2 q0, h0;")
add("e2m1x2.e5m2x2", "cvt.rn.satfinite.e2m1x2.e5m2x2 q0, h0;")
# fp6 <-> fp8
add("e4m3x2.e3m2x2", "cvt.rn.satfinite.e4m3x2.e3m2x2 h0, h0;")
add("e3m2x2.e4m3x2", "cvt.rn.satfinite.e3m2x2.e4m3x2 h0, h0;")
add("e4m3x2.e2m3x2", "cvt.rn.satfinite.e4m3x2.e2m3x2 h0, h0;")
# controls: the converts that DO exist (fp4<->f16x2), all archs
add("CTRL.f16x2.e2m1x2", "cvt.rn.f16x2.e2m1x2 hx0, q0;")
add("CTRL.e2m1x2.f16x2", "cvt.rn.satfinite.e2m1x2.f16x2 q0, hx0;")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes_conv.json")
json.dump(P, open(out, "w"), indent=1)
print(len(P), "probes ->", out)
