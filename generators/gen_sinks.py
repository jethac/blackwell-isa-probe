#!/usr/bin/env python3
"""Gap-1 SASS confirmation: emit real load->op->store kernels for the u8x4/s8x4
ops so cuobjdump --dump-sass shows the actual hardware lowering (single SIMD
instruction vs a PRMT/emulation trampoline). One .ptx per op; caller runs
ptxas -arch=sm_120a -O3 then cuobjdump -sass. Written for whichever ops the
accept/reject sweep marks ACCEPT on consumer; harmless if an op is rejected
(that .cubin just won't build).
"""
import os, sys

SINK = """\
.version 9.2
.target {target}
.address_size 64

.visible .entry sink(.param .u64 po, .param .u64 pa, .param .u64 pb)
{{
    .reg .b64 rd<4>;
    .reg .b32 r<4>;
    ld.param.u64 rd0, [po];
    ld.param.u64 rd1, [pa];
    ld.param.u64 rd2, [pb];
    cvta.to.global.u64 rd0, rd0;
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    ld.global.u32 r1, [rd1];
    ld.global.u32 r2, [rd2];
    {code}
    st.global.u32 [rd0], r0;
    ret;
}}
"""

# name -> instruction (binary ops use r1,r2 -> r0; unary uses r1 -> r0)
OPS = {
    "add_u8x4":     "add.u8x4 r0, r1, r2;",
    "sub_u8x4":     "sub.u8x4 r0, r1, r2;",
    "min_u8x4":     "min.u8x4 r0, r1, r2;",
    "max_u8x4":     "max.u8x4 r0, r1, r2;",
    "min_s8x4":     "min.s8x4 r0, r1, r2;",
    "max_s8x4":     "max.s8x4 r0, r1, r2;",
    "add_s8x4":     "add.s8x4 r0, r1, r2;",
    "neg_s8x4":     "neg.s8x4 r0, r1;",
    "add_sat_u32":  "add.sat.u32 r0, r1, r2;",
    "add_sat_s16x2":"add.sat.s16x2 r0, r1, r2;",
}

target = sys.argv[1] if len(sys.argv) > 1 else "sm_120a"
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "sinks")
os.makedirs(outdir, exist_ok=True)
for name, code in OPS.items():
    with open(os.path.join(outdir, f"sink_{name}.ptx"), "w") as f:
        f.write(SINK.format(target=target, code=code))
print("wrote", len(OPS), "sink probes to", outdir, "target", target)
