# blackwell-isa-probe

**A compatibility matrix for the Blackwell PTX ISA, decided by `ptxas`, one single-instruction kernel at a time — from the tensor-core surface a serving-kernel author cares about out to a systematic pass over the documented PTX instruction set.**

993 single-instruction PTX probes × 6 Blackwell targets ≈ 5,958 `ptxas` invocations. Every cell is an assemble-or-arch-reject verdict from the toolchain that builds the binaries, with the load-bearing accepts disassembled to confirm real hardware lowering. The raw verdict matrix is [`results/results_full.json`](results/results_full.json); this repo is the generator, the harness, and the data behind it.

> Companion to the article **"Which tensor-core instructions does each Blackwell chip have? I had Claude ask `ptxas` 296 times."**
> *(Link forthcoming — the article isn't published yet.)*

## From a curated table to the whole instruction set

This started as a **curated 296-probe table**: the tensor-core / memory / sync surface a serving-kernel author actually touches (block-scaled FP4/FP6 matrix-multiply, `tcgen05`, the async-copy/TMA machinery, clusters, warp reductions), built as the companion to the article above. That curated set is preserved here in full.

It has since been **expanded to a comprehensive sweep of the documented PTX instruction set** — integer and extended-precision arithmetic, the whole single/double/half floating-point surface, comparison and selection, logic and shift, bit manipulation, data movement and the full conversion matrix, texture and surface, control flow, barriers and fences, the atomic family, warp-level primitives, the scalar and SIMD video instructions, and the special registers. The goal is a single authoritative answer to "does *this* instruction exist on *this* Blackwell target," not just for the tensor path but across PTX. Started curated; now exhaustive (to the limits stated under **Coverage**).

The payoff of going wide is in the last section of the findings: it tells you which parts of the ISA *don't* split, which is exactly what a portable kernel needs to know.

---

## Why

The answer isn't consolidated in one place. There's no single published table of which instructions exist on which Blackwell chip; the PTX ISA manual carries it across hundreds of "Target ISA Notes" paragraphs written in a family-suffix algebra (`sm_120a` vs `sm_120f` vs "sm_100 or higher"). The CUDA Programming Guide's feature table says FP4/FP6 tensor cores are "Yes" for datacenter and consumer alike — and that's accurate; the nuance is that the two camps reach FP4 through *different instruction families*, so a "Yes" in the datacenter column and a "Yes" in the consumer column are not the same instruction. The documentation is accurate with nuance; this repo just makes the nuance mechanical to look up.

There is a definitive check, and it is the thing that actually builds the binaries: **`ptxas`**. Feed it a minimal kernel containing exactly one instruction, target each chip, and it either assembles or reports `not supported on .target`. Do that 993 times against six targets, triage every reject by error text (an arch reject is a real ISA answer; a syntax/operand/version reject is a harness artifact and must not be counted as one), and disassemble the interesting accepts to confirm the SASS is real hardware and not a software trampoline. That is this repo.

## The six targets

| Column | Target | Chip |
|---|---|---|
| `sm_100a` | B200 | datacenter Blackwell |
| `sm_103a` | B300 | datacenter Blackwell (Ultra) |
| `sm_110a` | Thor | automotive/robotics SoC |
| `sm_120a` | RTX 50-series (e.g. 5090, RTX PRO 6000) | consumer Blackwell |
| `sm_120f` | the cc-12.x **family** target | consumer Blackwell, one-binary family build |
| `sm_121a` | DGX Spark (GB10) | consumer Blackwell in a desk box |

## Headline findings

**The family splits along exactly one line: which tensor-core instruction set the chip speaks — and it is *not* datacenter-vs-consumer.** A quick glossary for the terms used below: **`mma.sync`** is the warp-level tensor-core matrix-multiply-accumulate instruction; **`tcgen05`** is the datacenter-Blackwell tensor-core MMA family that computes through a dedicated on-chip *tensor memory* rather than warp registers; **TMA** (the `cp.async.bulk.tensor` instructions) is the Tensor Memory Accelerator, hardware for bulk multidimensional global↔shared copies; a **cluster** is a group of cooperating thread blocks that can read each other's shared memory; **`ldmatrix`** loads a matrix tile from shared memory into the registers `mma.sync` expects; an **`mbarrier`** is an in-shared-memory arrival barrier used to signal completion of async copies; **`multimem`** is the NVLink multicast/reduce load-store family; **`redux.sync`** is a single-instruction warp-wide reduction.

- **Two camps.** *Camp `tcgen05`* = **B200, B300, and Thor** — tensor memory, `tcgen05.mma`, paired-CTA `cta_group::2`, block-scaled FP4 through `tcgen05`. *Camp warp-block-scale* = **RTX 50, the 120f family target, and Spark** — everything FP4/FP6 lives in `mma.sync` with the `kind::mx*` modifiers. Each camp arch-rejects the other's entire tensor path, instruction for instruction. Thor is a datacenter-lineage chip in a robot; Spark is a 5090 in a desk box. The form factor tells you nothing; the tensor lineage tells you everything.
- **Spark == 5090.** On all 993 instructions co-probed on both, **zero differences**. For kernel authors, a DGX Spark is a 5090 with unified memory and fewer SMs.
- **`u8x4`/`s8x4` SIMD integer is warp-block-scale-camp-wide** (`sm_120a`, `sm_120f`, `sm_121a`), not "120f-only" as first mined — and lowers to real `VIADD.U8`/`VIMNMX.U8` byte-lane SASS, not a `PRMT` emulation trampoline. (See footnote 7.)
- **3-input min/max splits by type** (a SASS-level fusion, not a PTX instruction). Float `FMNMX3` fuses across the **entire `tcgen05` camp including Thor**; integer `VIMNMX3` is present on B200/B300 but **absent on Thor**; consumer Blackwell has neither and pays the 2-input cost in the softmax running-max. (Footnote 8.)
- **Consumer clusters launch at runtime.** The compile oracle can't see the runtime, so for the one row where it mattered we ran the experiment ([`probes/j_cluster_probe.cu`](probes/j_cluster_probe.cu)): on an RTX PRO 6000, multi-CTA clusters launch, distributed shared memory works (one rank reads another's shared memory and gets the exact bytes), and `cluster.sync()` is load-bearing (a no-barrier control races). The only real limit is *size*: 8 portable everywhere, 16 on `sm120` via the non-portable flag, 8 on Spark.
- **`.multicast::cluster` TMA: accepted, but unicast-lowered on consumer.** It assembles *with a `ptxas` advisory warning* and lowers to a plain **unicast** `UTMALDG.2D` — byte-identical to the non-multicast form, versus real multicast on B200. So compile-acceptance here does not imply a hardware multicast fast-path. ([`probes/j_multicast_tma.cu`](probes/j_multicast_tma.cu).)
- **Stochastic rounding (`cvt.rs`) is B200/B300 only** — Thor rejects it with the same error as the 5090. So does `redux.sync.{min,max}.f32` (the float warp-reduce). Both are datacenter-Blackwell-exclusive within the six targets, and both are **narrower than the tensor-core split** — they are present on two of the three `tcgen05`-camp chips, not all three.
- **The 120f family target loses exactly two instructions vs the 5090** — the *sparse* block-scaled FP4 MMAs (`a`-suffix-exclusive). So dense NVFP4 kernels compile once for `sm_120f` and run on the 5090, the RTX PRO 6000, and Spark from one binary; the belief that the family target "can't do NVFP4 MMA" is true only for the sparse variant.
- **B300 vs B200: zero differences** across everything probed.

## What the comprehensive sweep adds

Going from the curated tensor/memory/sync surface to the full instruction set (689 additional probes) asks a different question: **does the differentiation continue outside the tensor path, or is the rest of PTX a common baseline?** The answer is clean, and it is the useful part:

- **Of the 689 comprehensive probes, 684 assemble identically on all six targets.** Integer and extended-precision arithmetic; the entire single- and double-precision floating-point surface including `rcp`/`sqrt`/`rsqrt`/`sin`/`cos`/`lg2`/`ex2`/`tanh` and every rounding mode; FP16/BF16 (scalar and packed) arithmetic; comparison/selection; logic/shift; bit manipulation (`popc`/`clz`/`bfind`/`brev`/`bfe`/`bfi`/`szext`/`bmsk`/`prmt`); the data-movement and conversion matrix (`ld`/`st` across state spaces, cache operators and memory scopes, `cvt` across int/float/`tf32`); texture and surface; control flow; barriers, fences and `membar`; the atomic family (`add`/`min`/`max`/`and`/`or`/`xor`/`cas`/`inc`/`dec` across global/shared, all scopes, and the `f16`/`bf16`/`f32`/`f64`/vector/`b128` types); the warp-level primitives (`vote.sync`/`match.sync`/`shfl.sync`/`activemask`/`redux.sync` integer); the scalar and SIMD video instructions; and the special registers — **all universal across B200, B300, Thor, RTX 50, the 120f target, and Spark.**
- **Exactly one comprehensive-family probe splits across targets**, and it is not a new phenomenon: `add.sat.s16x2` — the *saturating* packed-16-bit integer add — is warp-block-scale-camp-only, the same PTX 9.2 saturating-SIMD family as the `u8x4`/`s8x4` ops in footnote 7. The **non-saturating** packed forms (`add`/`sub`/`min`/`max.s16x2`) are universal; only the saturating variants split. On consumer it lowers to a genuine `VIADD.S16x2.ISAT` (SASS-confirmed), not an emulation.
- **A handful of probes are rejected on all six** — not an arch difference but a fact about the ISA: the deprecated non-`.sync` warp instructions (`vote.all`, `vote.ballot`, `shfl.up` without `.sync`) are gone on every Blackwell target, `.oob` (out-of-bounds) `fma` is a half/bhalf-only modifier (not `f32`), and there is no packed `.u8x4` abs-diff in either camp.

The practical reading for a kernel author: **portability guards are only needed for the five families the curated sweep already mapped** — the tensor-core MMA path, `cvt.rs`, the float `redux.sync`, saturating packed-SIMD, and sparse block-scale. The rest of PTX is one Blackwell baseline. The comprehensive sweep's contribution is that *confirmed common baseline*, made mechanical to re-check.

## The matrix

Legend: ✅ assembles · ❌ arch-reject · ⚠️ assembles, but read the footnote

| Instruction family | B200 (100a) | B300 (103a) | Thor (110a) | RTX 50 (120a) | 120f family | Spark (121a) |
|---|---|---|---|---|---|---|
| `mma.sync` f16/bf16/tf32/int (baselines) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mma.sync` fp8 (e4m3/e5m2, f32 & f16 acc) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mma.sync` **fp6/fp4** (`kind::f8f6f4`, all 21 mixes) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `mma.sync` block-scaled **nvfp4** (`kind::mxf4nvf4`, the consumer block-scaled FP4 MMA) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `mma.sync` block-scaled mxfp4 / mxfp8-fp6-fp4 (`kind::mxf4` / `mxf8f6f4`) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Sparse** block-scaled fp4 (`mma.sp::ordered_metadata.kind::mxf4nvf4` k128) | ❌ | ❌ | ❌ | ✅ | **❌** | ✅ |
| `tcgen05.*` — datacenter-Blackwell tensor-core MMA via dedicated *tensor memory* | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `tcgen05.mma...block_scale` (datacenter's NVFP4 path) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `wgmma.*` — warp-group MMA, the Hopper-generation tensor-core instruction | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ ¹ |
| `cvt` packed fp8/fp6/fp4 ↔ f32/f16x2/bf16x2 (the full narrow-float convert set) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cvt.rs.*` — **stochastic rounding** | ✅ | ✅ | **❌** | ❌ | ❌ | ❌ |
| `ldmatrix` (shared→register tile load) m8n8 b16, `stmatrix`, `movmatrix` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ldmatrix` sub-byte: `b4x16_p64` (fp4), `b6x16_p32` (fp6), m16n16 b8 trans | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cp.async.bulk` + full **TMA** (bulk multidim copy: `bulk.tensor` 1d–5d, im2col, prefetch) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ² |
| TMA `.multicast::cluster` | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ ³ |
| `cp.async.bulk.tensor...cta_group::2` (paired-CTA TMA) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `mbarrier.*` incl. `expect_tx` (the async-copy arrival barrier) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Clusters: `barrier.cluster`, `mapa`, `ld/st.shared::cluster`, distributed shared memory | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ⁴ |
| `redux.sync` (warp-wide reduce, integer) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `redux.sync.{min,max}.f32` (float warp-reduce) | ✅ | ✅ | **❌** | ❌ | ❌ | ❌ |
| `multimem.ld_reduce` (NVLink multicast / symmetric-memory loads) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ⁵ |
| `griddepcontrol.*` (programmatic dependent launch) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `setmaxnreg` (register-count hint) | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ ⁶ |
| **Integer / float / half arithmetic**, `cmp`/`sel`, logic/shift, bit-manip, `cvt` int/float/tf32 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Data movement** (`ld`/`st` all state spaces + cache ops + scopes + vector widths) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Atomics** (`atom`/`red`: add/min/max/and/or/xor/cas/inc/dec; f16/bf16/f32/f64/vector/b128) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Warp-level** (`vote.sync`/`match.sync`/`shfl.sync`/`activemask`), **texture/surface**, control flow | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Non-saturating packed-int SIMD (`add`/`min`/`max.s16x2`), scalar+SIMD video | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Saturating** packed-int SIMD (`add.sat.s16x2`, `u8x4`/`s8x4` add/sub/min/max) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ ⁷ |
| 3-input min/max (`FMNMX3`/`VIMNMX3`, SASS fusion at -O3) | ✅ | ✅ | ⚠️ | **❌** | **❌** | **❌** ⁸ |

¹ `wgmma` (warp-group MMA) is `sm_90a`-only. It is on no Blackwell target, either camp — the Blackwell tensor-core work moved to `tcgen05` (datacenter) and `mma.sync` block-scaled (consumer).

² Real `TMALDG`/`TMASTG` SASS on consumer, byte-identical machinery to B200 — for `shared::cta` destinations. Cluster-destination copies carry a compiler-inserted guard on the remote-rank path.

³ Accepted on the 12.x targets with an actual `ptxas` *advisory warning*, and it lowers to a plain **unicast** `UTMALDG.2D` — byte-identical to the non-multicast form, zero `UTMALDG.MULTICAST` opcodes, versus real multicast on B200. So on consumer, compile-acceptance here does not imply a hardware multicast path — confirmed at the SASS level.

⁴ The entire cluster ISA assembles to genuine CGA hardware SASS on consumer (`CGABAR_ARV`/`CGABAR_WAIT`, same bytes as B200) — **and it launches at runtime.** On an RTX PRO 6000, multi-CTA clusters launch cleanly, distributed shared memory works (a rank reads another rank's shared memory through `cluster.map_shared_rank`, verified against a no-barrier race control), and the passing kernel disassembles to real `UCGABAR_ARV`/`UCGABAR_WAIT`. Portable cluster size 8 works across the consumer camp; `sm120` also takes 16 via `NonPortableClusterSizeAllowed`, while Spark (`sm121`) caps at 8.

⁵ Real `LDGMC` multicast-load SASS on the 5090, not emulation. `multimem.st`/`red` lower to plain stores on every arch including B200.

⁶ Assembles everywhere, compiles to zero SASS in a trivial kernel everywhere. Compile acceptance tells you nothing about this instruction on its own.

⁷ The PTX 9.2 saturating SIMD integer ops — `add`/`sub`/`min`/`max`/`neg` on `.u8x4`/`.s8x4`, saturating `add.sat`/`sub.sat` on `.u8x4`/`.s8x4`/`.s16x2`/`.u16x2`/`.u32` — assemble on the **whole warp-block-scale camp** and arch-reject on the entire `tcgen05` camp. "120f" is the *family* label (a feature promoted to `sm_120f` is exposed across the whole 12.x family, exactly like the dense NVFP4 MMA), not an exclusion of 120a/121a. The comprehensive sweep sharpens this: the **non-saturating** packed-16-bit ops (`add`/`sub`/`min`/`max.s16x2`) are universal on all six targets; only the **saturating** variants are camp-split. On `sm_120a` each lowers to one real SIMD instruction — `VIADD.U8x4.ISAT`, `VIADD.S16x2.ISAT`, `VIMNMX.U8`/`.S8` — genuine byte/half-lane hardware. (There is no packed `.u8x4` abs-diff in either camp; only the legacy `vabsdiff4.u32`, which assembles everywhere.)

⁸ Not a PTX instruction — a SASS fusion (chained `fmax`/`max` at -O3), probed on all six targets plus a Hopper (`sm_90a`) control ([`probes/fmnmx3.cu`](probes/fmnmx3.cu)). It splits by **type**. *Float* `FMNMX3` fuses across the **entire `tcgen05` camp** — B200, B300 **and Thor** — and on none of the consumer camp (120a/120f/121a emit two `FMNMX`). *Integer* `VIMNMX3` is narrower: Hopper, B200 and B300 have it, but **Thor does not** (it falls back to `IMNMX` pairs), and consumer has neither. So B200/B300 have both the float and integer fusion, **Thor has the float fusion only**, and consumer Blackwell has neither. The regression is a consumer-camp fact, byte-identical on 120a, 120f *and* 121a.

**Want the raw verdicts?** Every cell above is derived from [`results/results_full.json`](results/results_full.json). Regenerate the matrix, the cross-target deltas, and the camp summary straight from that data — nothing hand-typed — with:

```
python results/summarize.py
```

## How to run

**Requirements:** a CUDA 13.2 toolkit on `PATH` (`ptxas`, `nvcc`, `cuobjdump`, `nvdisasm`) and Python 3. No GPU is needed for the compile probes (`ptxas` is a cross-assembler); the cluster runtime probe ([`probes/j_cluster_probe.cu`](probes/j_cluster_probe.cu)) does need a consumer Blackwell card.

Comprehensive sweep — generate the full manifest, then run the oracle over all six targets:

```
python generators/gen_probes_full.py                          # -> generators/probes_full.json (993 probes)
python harness/runner.py generators/probes_full.json results/results_full.json
```

`gen_probes_full.py` folds in the curated generators (`gen_probes.py`, `gen_conv_probes.py`, `gen_gap_probes.py`) and adds the comprehensive families on top, so one command produces the whole manifest. `runner.py` wraps each snippet in a minimal `.ptx` (`.version 9.2`, the highest ptxas 13.2 accepts; auto-upgraded from a lower floor if a target needs it), runs `ptxas -arch=<target>` for each arch in `ARCHES`, and records `ACCEPT` / `REJECT(class)`. It applies the PTX-version floor for Thor (`sm_110a` needs PTX ISA ≥ 9.0) and triages every reject by error text — `ARCH` (a real ISA answer) is kept separate from `SYNTAX`/`OPERAND-TYPE`/`PTXVER` (harness artifacts, never counted as arch answers). `harness/extend_archs.py` records how the datacenter/Thor columns were first added to the curated `results.json`, including the version-floor fix.

Just the curated 296-probe companion sweep (the article's set):

```
python generators/gen_probes.py                     # -> generators/probes.json (258 core probes, families 1-7)
python harness/runner.py generators/probes.json     # -> results.json
python generators/gen_gap_probes.py                 # u8x4/s8x4 SIMD int (footnote 7)
python generators/gen_conv_probes.py                # fp4<->fp8/fp6 packed converts
python generators/gen_sinks.py sm_120a              # load->op->store kernels for SASS confirmation
```

## Coverage

This repo covers the **documented PTX ISA instruction set as implemented by `ptxas` 13.2** (which accepts PTX ISA up to `.version 9.2`), probed across all six Blackwell targets. **993 single-instruction probes** — 304 in the curated tensor/memory/sync families, 689 in a systematic pass over the rest of the documented "Instructions" chapters — × 6 targets = **5,958 `ptxas` invocations**.

**Sampling rule.** Type-parametrized instructions are probed across the meaningful type variants where a difference can hide (integer `s32`/`u32`/`s64`/`u64` plus a 16-bit sample; float `f32`/`f64`; half `f16`/`f16x2`/`bf16`/`bf16x2`; bitwise `b16`/`b32`/`b64`), each rounding mode sampled at least once per instruction that carries one, `ftz`/`sat` sampled on `f32`, and one representative state space / cache operator / memory scope per `ld`/`st`/`atom` — **not** the full cross product. This is a representative-variant sweep, not every permutation; where a whole type family matters (the 21 `kind::f8f6f4` FP6/FP4 mixes, the 25 `mxf8f6f4` A×B mixes) it is enumerated in full.

**Known gaps**, stated plainly:
- Instructions introduced only in **PTX ISA 9.3+** (for example the `fabric.*` distributed-memory family) are outside this `ptxas` snapshot by construction — a probe for them reports "unknown instruction" on all six targets, which says nothing about the targets.
- **`brx.idx`** (indexed branch) is not probed: it needs a `.branchtargets` table and the labels it indexes, which the single-instruction harness does not model.
- **Texture/surface** is probed in the unified/bindless (register-handle) form only.
- The SASS-dump sink duplicates (curated family `9_sink`) and a couple of legacy video spellings are documented in the curated data and prose rather than re-emitted in `results_full.json`.

The claim is precise: *this covers the documented PTX 9.2 instruction set as probed under `ptxas` 13.2, at the sampling density above.* It does not claim every permutation of every modifier.

## Caveats

- **This is `ptxas` 13.2's worldview.** The results are a **version snapshot**: `ptxas` **V13.2.78** for the full `results_full.json` sweep (and for the original curated `results.json`; `results/gap_results.json` was **V13.2.51**). Older toolchains reject things this one accepts (there are community reports of 12.8-era `ptxas` refusing dense NVFP4 on 120f); newer ones will differ again. A snapshot with a version number, not scripture.
- **Acceptance ≠ performance.** Everything relied on was disassembled to confirm real hardware lowering — that is how the multicast unicast-lowering and the zero-SASS `setmaxnreg` were caught. For compile-only rows, ✅ means "the ISA admits it," nothing about throughput.
- **The compile oracle can't see the runtime.** For the cluster row — where the method's honest limit met a decade of "5090s can't launch clusters" folklore — we ran the 20-line runtime experiment. Docs right, folklore wrong.
- **Six targets, one harness bug we caught and fixed.** The first Thor column was wrong: `sm_110` needs PTX ISA 9.0+ and the auto-version-upgrade didn't catch that specific error form, so cells recorded "rejected" that were really "the harness sent an old dialect." Re-probed and fixed (`extend_archs.py`, and a per-target version floor in `runner.py`) — a reminder that compile-probing lies most confidently about the target you added last. The comprehensive sweep triages every reject by error string precisely so that a syntax/operand/version artifact is never recorded as an arch answer.

## Repo layout

```
generators/
  gen_probes.py        the curated core: 258 single-instruction probes, families 1-7
  gen_gap_probes.py    u8x4/s8x4 SIMD int gap-fill (footnote 7)
  gen_conv_probes.py   fp4<->fp8/fp6 packed-convert probes
  gen_probes_full.py   the comprehensive generator: folds in the curated families and
                       adds the rest of the documented PTX instruction set (families 10-25)
  gen_sinks.py         load->op->store kernels for SASS confirmation
harness/
  runner.py            the ptxas oracle: wrap one instruction, assemble per target, record
                       the verdict; six-target ARCHES, per-target PTX-version floor, reject triage
  extend_archs.py      how the sm_103a/sm_110a columns were first added (+ the Thor floor fix)
  run_gap.sh           the gap-fill orchestrator (as-run record; flat-dir layout)
probes/
  fmnmx3.cu            3-input min/max SASS fusion probe (FMNMX3 / VIMNMX3)
  j_cluster_probe.cu   the cluster RUNTIME experiment (launch + DSM + cluster.sync)
  j_multicast_tma.cu   multicast-TMA SASS check (unicast-lowering confirmation)
results/
  results_full.json    the comprehensive 993-probe x 6-target verdict matrix (the value)
  results.json         the original curated 296-probe x 6-target matrix
  gap_results.json     the 39 gap-fill probes (SIMD int + converts), 6 targets
  summarize.py         regenerate the matrix + cross-target deltas + camp summary from the JSON
  j_probe_full_raw.txt the run log: box, ptxas version, probe count, deltas, coverage, commit
```

## Provenance

The probes, the harness, the six-target sweeps, the SASS verification, and this writeup were generated by **Claude** under a charter set by **Jetha Chan**, who owns the framing and the sign-off. The matrix is machine-generated and machine-checked — the raw `results_full.json` ships with it, and [`results/summarize.py`](results/summarize.py) regenerates every claim in the tables from that data. If a cell here disagrees with your toolchain, the JSON and the generator are right there to re-run.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Jetha Chan.
