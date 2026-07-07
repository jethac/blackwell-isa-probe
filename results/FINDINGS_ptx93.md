# PTX 9.3 / CUDA 13.3 on Blackwell — re-probe findings

*Companion to this repo's PTX 9.2 / ptxas 13.2 matrix, which flagged that "newer toolchains will differ again." This is the CUDA 13.3 / PTX 9.3 re-run — on both an x86 host and a real GB10 DGX Spark (arm64).*

## Method
- Toolchain: `ptxas` **V13.3.73** (pip `nvidia-cuda-nvcc==13.3.73`) on x86 (Windows) **and** aarch64 (GB10 Spark). Both accept `.version 9.3` (ceiling 9.3).
- Re-ran the full **993-probe** sweep at 9.3 and diffed vs the 9.2 matrix (`results/diff_9293.py`).
- Authored a **19-probe PTX 9.3 delta** (`generators/gen_probes_ptx93.py`), grammar verified against the **PTX ISA 9.3 manual** *and* the live oracle.
- SASS-gated the door-relevant accepts (`cuobjdump`). Ran everything on both hosts.

## Result 1 — the 9.2 matrix is stable across the bump
**0 accept↔reject flips, 0 reject-class changes** across all 993 probes × 6 targets, 13.2→13.3. ptxas 13.3 did not re-classify a single documented-9.2 instruction on any Blackwell target. All 9.3 novelty is additive.

## Result 2 — the 9.3 additions are common baseline, not a new split
All 19 spec-verified 9.3 probes **ACCEPT on all six targets** (100a/103a/110a/120a/120f/121a), and are **byte-identical x86 ≡ arm64**. None splits the consumer vs datacenter camps — the manual's own "Requires sm_90 or higher" notes agree. Converged families: `fabric.submit`/`fabric.wait`, async `multimem.st/red`, `red.async` (sys-only), `fence.proxy.fabric::generic.alias`, `clmad`, the `mma_throughput` pragma, and the extended `mbarrier` set (`.phase_type::primary`, `check_layout.layout::{v0,v1}`, `init.layout::v1`, `pending_count.layout::v0`).

## Result 3 — SASS gate (accepted ≠ implemented)
- **`fabric.try_get` splits by camp at the SASS level, invisibly to the compile oracle.** It ASSEMBLES on all six targets, but lowers to a **`BPT.TRAP`** (non-functional stub) on the entire **warp-block-scale camp** (sm_120a / 120f / sm_121a) and to a **real path** (`ELECT`/`R2UR`, no trap) on the **tcgen05 camp** (sm_100a / sm_103a / sm_110a). So consumer Blackwell compile-accepts the fabric data-movers but *traps* on them — fabric is datacenter-only in hardware, following the exact TMEM fault line. This is the disassemble-to-catch-the-trampoline discipline delivering the answer a runtime test would have — for free and more airtight. (`clmad` is universally miscompiled across *all* camps; fabric is the one that splits.)
- `fabric.submit` → **real SASS** on the GB10: `UTMACMDFLUSH ; DEPBAR.LE ; CCTL.IVALL` (a genuine flush/completion primitive; identical x86-built vs GB10-built). The operand-less `submit`/`wait` do not trap on consumer; the data-moving `try_*` ops do.
- `mma_throughput` pragma → **zero effect**: byte-identical SASS with/without it around a real `OMMA.SF.16864…` block-scale MMA on sm_121a.
- `clmad` → **miscompiled on all four Blackwell targets** (100a/103a/120a/121a): accepts, emits no carry-less arithmetic, stores the wrong operand (spec §9.7.1.5 says `clmul(a,b)+c`). Universal `ptxas` 13.3.73 bug, not camp-specific.

## Result 3b — the FULL SASS camp-split sweep (all 23 delta probes)
Every 9.3 probe assembles on all six targets, but disassembling each on a datacenter (`sm_100a`) vs a consumer (`sm_121a`) target shows **8 of 23 split by camp — in two distinct modes**, both invisible to the compile oracle:

| Mode | Probes | Consumer (sm_120/121) | Datacenter (sm_100/103/110) |
|---|---|---|---|
| **hard trap** | `fabric.try_get`, `try_put`, `try_put.multimem`, `try_red` | **`BPT.TRAP`** (non-functional stub) | real path (`ELECT`/`R2UR`/`UBLKCP`/`SYNCS`) |
| **degraded fallback** | `multimem.st.async`, `multimem.red.async`, `red.async.sys` (add/min) | `MEMBAR.ALL.SYS ; ERRBAR ; CGAERRBAR ; REDG…SYS` — the async is compiled away to a **synchronous** reduce behind error barriers | true async path: warp-`ELECT` loop around `UREDGR…SYS` (uniform async reduce) |
| identical | the other 15 (`fabric.submit`/`wait`, `fence.proxy.fabric`, `clmad`, `mma_throughput`, all 7 `mbarrier`) | — same SASS both camps — | |

Read: the **entire 9.3 multi-GPU / async-fabric family** (`fabric.try_*` + async `multimem`/`red.async`) is a datacenter capability. Consumer Blackwell compile-accepts all of it, then either **traps** (fabric data-movers) or **silently drops the async** and runs a synchronous fallback (multimem/red.async). A fork author guarding by "does it assemble on `sm_120a`?" gets "yes" on every one — and ships a `BPT.TRAP` or a non-async reduce. This is the guard-the-right-*kind*-of-limit lesson, extended to the whole 9.3 async surface, with SASS receipts.

## Is there a new consumer-Blackwell kernel opportunity in 9.3?
Scored by *native-on-consumer × library-skipped × serving-critical*:

| 9.3 item | Consumer? | Serving-path? | Verdict |
|---|---|---|---|
| `mma_throughput` pragma | yes (all 6) | tensor path | **No-op** on the block-scale MMA at codegen. The only 9.3 item that even touches the serving path, and it moves nothing. (Runtime-scheduler effect with identical SASS is unlikely; Spark-checkable.) |
| `mbarrier` `.phase_type`/`.layout`/`check_layout` | yes (all 6) | async pipeline | Plumbing refinements; no new capability. `red.async.release` is **sys-only** (ptxas + spec agree) → does *not* enable the cluster-scope split-KV DSM-reduce lever. |
| `fabric.*` | assembles all 6, **but `try_*` traps on consumer** | **multi-GPU** | **Hard non-door** on consumer — the data-movers lower to `BPT.TRAP` on the whole warp-block-scale camp (real only on tcgen05). Datacenter-only in hardware. `submit`/`wait` are real flush/wait primitives on both. |
| `clmad` | yes (all 6) | off-path (GF2 crypto) | Miscompiled; irrelevant to serving. |
| Tile C++ scaled-MMA / i4 / fp4 (**not probed**) | ? | tensor path | The one thing worth a follow-on: a *library* surface for the consumer block-scale MMA. Separate NVRTC probe. |

**Verdict: no new consumer-Blackwell kernel door in the PTX 9.3 delta.** The novelty is real but aimed at multi-GPU fabric/collectives and crypto — not consumer-Blackwell serving kernels. The existing warp-level block-scale MMA path is still where the value is; 9.3 opened no new one on this silicon.

## Two ptxas 13.3.73 bugs (found in passing, filed with NVIDIA + public repros)
1. **`clmad` miscompile** — on all four Blackwell targets (sm_100a/103a/120a/121a), `clmad.{lo,hi}.u64` accepts but emits no carry-less arithmetic and stores an input operand instead of `clmul(a,b) ^ c` (spec §9.7.1.5). No prior report found — apparently novel. Repro: https://github.com/jethac/ptxas-clmad-miscompile
2. **Internal compiler error (C7907)** on `red.async.release.sys.global.add.u32 [a], v, [mbar];` (sys form with a trailing mbarrier operand); the no-mbar form assembles fine. C7907 is a known Blackwell ICE class (NVIDIA/numba-cuda#725, state-spaces/mamba#904, triton-lang/triton#9933 — all large Triton kernels); this is a single-instruction, register-pressure-free repro. Repro: https://github.com/jethac/ptxas-red-async-c7907-ice

## Artifacts
- `results/results_full_ptx93.json` (x86 9.3) / `results/results_full_ptx93_arm.json` (GB10 9.3) — full matrices.
- `results/results_ptx93.json` / `_arm.json` — the 19-probe delta, both hosts.
- `results/diff_9293.py` — the diff tool.
- `generators/gen_probes_ptx93.py` — the spec-verified 9.3 delta generator.
- `harness/runner.py` — `MAX_PTX_VERSION 9.3`, `PTXAS` env pin (9.2-vs-9.3 side-by-side).
