#!/usr/bin/env python3
"""PTX ISA 9.3 delta probes (CUDA 13.3 / ptxas 13.3.x).

The comprehensive sweep in gen_probes_full.py is a ptxas-13.2 / PTX-9.2 snapshot;
that toolchain hard-caps at .version 9.2 and reports every 9.3 instruction as
"unknown instruction" on all six targets (a toolchain-missing artifact, NOT an
arch answer). This generator adds the instruction families PTX ISA 9.3
introduced, run under ptxas 13.3.x (which accepts .version 9.3). It is a DELTA:
run it in addition to probes_full.json, then diff.

    PTXAS='.../cu13/bin/ptxas[.exe]' python harness/runner.py \
        generators/probes_ptx93.json results/results_ptx93.json

GRAMMAR SOURCE. Every form below is verified against the **PTX ISA 9.3 manual**
(docs.nvidia.com/cuda/pdf/ptx_isa_9.3.pdf) AND confirmed to assemble on the live
ptxas 13.3.73 oracle. The manual pins the operand grammar (e.g. clmad is
`clmad.mode.u64 d,a,b,c`, u64-only, mode={.lo,.hi}, sec 9.7.1.5; check_layout is
`mbarrier.check_layout.layout::{v0,v1}.shared::cta.b64 p,[a]`, sec 9.7.14.16.21;
fabric.wait takes NO register operands, only `.sync_restrict::reads`, sec
9.7.10.5.6). Verdicts are real ISA answers, not harness artifacts.

HEADLINE: every converged 9.3 form assembles IDENTICALLY on all six targets
(100a/103a/110a/120a/120f/121a) -- confirmed on BOTH an x86 and the GB10 arm64
host. None of the 9.3 additions splits the consumer vs datacenter camps; the
spec's own "Requires sm_90 or higher" notes agree. Whether an ACCEPT lowers to
real SASS vs a no-op/trampoline is Lane B (disassemble): fabric.submit -> real
UTMACMDFLUSH/DEPBAR/CCTL.IVALL; clmad + the mma_throughput pragma -> zero SASS in
a trivial kernel (and clmad MISCOMPILES -- see results notes).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ALL6 = ["sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_120f", "sm_121a"]
V93 = "9.3"

P = []
def add(family, name, code, **kw):
    d = {"family": family, "name": name, "code": code, "version": V93}
    d.update(kw)
    P.append(d)

SDECL = ".shared .align 16 .b8 sbuf[4096];"

# ---- 30_fabric : distributed-memory family (sec 9.7.10) ----------- #
# fabric.wait takes NO operands (only .sync_restrict::reads); fabric.submit is
# operand-less (optionally .op_restrict::fetching). The try_* data-movers need
# the full chain + typed operands (srcLeId is u32, srcDataOff is u64, size u32).
# ALL assemble on all six targets -- BUT the try_* ops are the one 9.3 family
# that SPLITS BY CAMP AT THE SASS LEVEL: they lower to a real path (ELECT/R2UR)
# on the tcgen05 camp (100a/103a/110a) and to BPT.TRAP (a non-functional stub) on
# the whole warp-block-scale camp (120a/120f/121a). Multi-GPU RMA; datacenter-only
# in hardware -- consumer compile-accepts then traps. Disassemble to see it.
F = "30_fabric"
add(F, "fabric.submit", "fabric.submit;")
add(F, "fabric.submit.op_restrict::fetching", "fabric.submit.op_restrict::fetching;")
add(F, "fabric.wait.sync_restrict::reads", "fabric.wait.sync_restrict::reads;")
add(F, "fabric.try_get",
    "fabric.try_get.async.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.b128 [rd0], [r0, rd1], r1, [rd2];", decls=SDECL)
add(F, "fabric.try_put",
    "fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 [r0, rd1], [rd0], r1, [rd2];", decls=SDECL)
add(F, "fabric.try_put.multimem",
    "fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 [r0, rd1], [rd0], r1, [rd2];", decls=SDECL)
add(F, "fabric.try_red.add.u32",
    "fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u32 [r0, rd1], [rd0], r1, [rd2];", decls=SDECL)

# ---- 31_multimem_async : async NVLink multicast/reduce ------------ #
# Requires .release; sys-scope (cta/cluster rejected); integer-add only.
F = "31_multimem_async"
add(F, "multimem.st.async.release.sys.f32",
    "multimem.st.async.release.sys.global.f32 [rd0], f0;")
add(F, "multimem.red.async.release.sys.add.u32",
    "multimem.red.async.release.sys.global.add.u32 [rd0], r0;")

# ---- 33_async_scope : red.async at sys scope (9.3 clarification) --- #
# red.async.release is SYS-ONLY -- ptxas rejects .cluster ("Illegal modifier
# '.cluster' … with '.release'"), directly confirming the 9.3 note. This limits
# the split-KV DSM-reduce lever (that wanted cluster-scope async reduce).
F = "33_async_scope"
add(F, "red.async.release.sys.add.u32",
    "red.async.release.sys.global.add.u32 [rd0], r0;")
add(F, "red.async.release.sys.min.s32",
    "red.async.release.sys.global.min.s32 [rd0], r0;")

# ---- 34_fence_fabric : fence.proxy fabric proxykind --------------- #
# .alias is mandatory with .fabric::generic.
F = "34_fence_fabric"
add(F, "fence.proxy.fabric.release.sys",
    "fence.proxy.fabric::generic.alias.release.sys;")
add(F, "fence.proxy.fabric.acquire.sys",
    "fence.proxy.fabric::generic.alias.acquire.sys;")

# ---- 35_clmad : carry-less multiply-add, GF(2) (sec 9.7.1.5) ------ #
# Spec: clmad.mode.u64 d,a,b,c ; mode={.lo,.hi}; ALL operands unsigned 64-bit.
# NOTE: assembles but MISCOMPILES on sm_121a (13.3.73) -- emits no carry-less
# arithmetic, stores the wrong operand. Reportable. Off the serving path.
F = "35_clmad"
add(F, "clmad.lo.u64", "clmad.lo.u64 rd0, rd1, rd2, rd3;")
add(F, "clmad.hi.u64", "clmad.hi.u64 rd0, rd1, rd2, rd3;")

# ---- 36_pragma : mma_throughput (compile-time NO-OP on block-scale MMA) ---- #
F = "36_pragma"
add(F, "pragma.mma_throughput", '.pragma "mma_throughput";')

# ---- 37_mbarrier93 : mbarrier machinery new/extended in 9.3 -------- #
# .phase_type::{primary,conditional} (sec 9.7.14.16.19); check_layout new in 9.3
# (sec .21); .layout::{v0,v1} on init/pending_count. All "Requires sm_90+".
F = "37_mbarrier93"
add(F, "mbarrier.try_wait.phase_type::primary",
    "mbarrier.try_wait.phase_type::primary.shared::cta.b64 p0, [rd0], rd1;", decls=SDECL)
add(F, "mbarrier.test_wait.phase_type::primary",
    "mbarrier.test_wait.phase_type::primary.shared::cta.b64 p0, [rd0], rd1;", decls=SDECL)
add(F, "mbarrier.try_wait.parity.phase_type::primary",
    "mbarrier.try_wait.parity.phase_type::primary.shared::cta.b64 p0, [rd0], r0;", decls=SDECL)
add(F, "mbarrier.check_layout.layout::v1",
    "mbarrier.check_layout.layout::v1.shared::cta.b64 p0, [rd0];", decls=SDECL)
add(F, "mbarrier.check_layout.layout::v0",
    "mbarrier.check_layout.layout::v0.shared::cta.b64 p0, [rd0];", decls=SDECL)
add(F, "mbarrier.init.layout::v1",
    "mbarrier.init.layout::v1.shared::cta.b64 [rd0], r0;", decls=SDECL)
add(F, "mbarrier.pending_count.layout::v0",
    "mbarrier.pending_count.layout::v0.b64 r0, rd1;", decls=SDECL)

# All 9.3 families targeted here are now converged (grammar verified vs the PTX
# ISA 9.3 manual + live ptxas 13.3.73). The `fabric.try_pullred` variant and the
# `.counted::bytes`/`.cp_mask` completion forms are the only remaining fabric
# spellings not enumerated (analogous; §9.7.10.5.3-4) -- low priority (the camp
# split is already established by try_get's BPT.TRAP-on-consumer SASS).

out = os.path.join(HERE, "probes_ptx93.json")
with open(out, "w") as f:
    json.dump(P, f, indent=1)
print(f"{len(P)} spec-verified PTX 9.3 delta probes written to {out}")
