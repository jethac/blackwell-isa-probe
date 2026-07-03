# blackwell-isa-probe

**A consolidated table of which tensor-core instructions each Blackwell chip actually has — decided by `ptxas`, one single-instruction kernel at a time.**

296 single-instruction PTX probes × 6 Blackwell targets ≈ 1,700 `ptxas` invocations. Every cell is an assemble-or-arch-reject verdict from the toolchain that builds the binaries, with the load-bearing accepts disassembled to confirm real hardware lowering. The raw verdict matrix is [`results/results.json`](results/results.json); this repo is the generator, the harness, and the data behind it.

> Reproducibility companion to the article **"Which tensor-core instructions does each Blackwell chip have? I asked `ptxas` 296 times."**
> *(Link forthcoming — the article isn't published yet.)*

---

## Why

The answer isn't consolidated in one place. There's no single published table of which tensor-core instructions exist on which Blackwell chip; the PTX ISA manual carries it across hundreds of "Target ISA Notes" paragraphs written in a family-suffix algebra (`sm_120a` vs `sm_120f` vs "sm_100 or higher"). The CUDA Programming Guide's feature table says FP4/FP6 tensor cores are "Yes" for datacenter and consumer alike — and that's accurate; the nuance is that the two camps reach FP4 through *different instruction families*, so a "Yes" in the datacenter column and a "Yes" in the consumer column aren't the same instruction. Community folklore fills the gaps, and it's often wrong in both directions.

But there's a definitive check, and it's the thing that actually builds the binaries: **`ptxas`**. Feed it a minimal kernel containing exactly one instruction, target each chip, and it either assembles or tells you `not supported on .target`. Do that 296 times against six targets, triage every reject by error text, and disassemble the interesting accepts to check the SASS is real hardware and not a software trampoline. That's this repo.

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

**The family splits along exactly one line: which tensor-core ISA the chip speaks — and it is *not* datacenter-vs-consumer.**

- **Two camps.** *Camp tcgen05* = **B200, B300, and Thor** — tensor memory, `tcgen05.mma`, paired-CTA `cta_group::2`, block-scaled fp4 through `tcgen05`. *Camp warp-block-scale* = **RTX 50, the 120f family target, and Spark** — everything fp4/fp6 lives in `mma.sync` with `kind::mx*`. Each camp arch-rejects the other's entire tensor path, instruction for instruction. Thor is a datacenter chip in a robot; Spark is a 5090 in a desk box. The SoC-ness tells you nothing; the tensor lineage tells you everything.
- **Spark == 5090.** On the 284 instructions co-probed on both in the main sweep, **zero differences** (and the 39-probe gap-fill adds zero more). For kernel authors, a DGX Spark is a 5090 with unified memory and fewer SMs.
- **`u8x4`/`s8x4` SIMD int is consumer-*camp*-wide** (`sm_120a`, `sm_120f`, `sm_121a`), not "120f-only" as first mined — and lowers to real `VIADD.U8`/`VIMNMX.U8` byte-lane SASS, not a PRMT emulation trampoline. (Corrected in the gap-fill; see footnote 7.)
- **3-input min/max splits by type.** Float `FMNMX3` fuses across the **entire tcgen05 camp including Thor**; integer `VIMNMX3` is present on B200/B300 but **absent on Thor** (Thor is float-only — the mirror image of Hopper's int-only). Consumer Blackwell has neither and pays the 2-input tax in the softmax running-max. (Corrected + extended in the gap-fill; see footnote 8.)
- **Consumer clusters launch at runtime.** The compile oracle can't see the runtime, so for the one row where it mattered we ran the experiment ([`probes/j_cluster_probe.cu`](probes/j_cluster_probe.cu)): on an RTX PRO 6000, multi-CTA clusters launch, distributed shared memory works (one rank reads another's smem and gets the exact bytes), and `cluster.sync()` is load-bearing (a no-barrier control races). The only real limit is *size*: 8 portable everywhere, 16 on `sm120` via the non-portable flag, 8 on Spark. Decade-old folklore that 5090s "can't launch clusters" is dead.
- **`.multicast::cluster` TMA: accepted, but unicast-lowered on consumer.** It assembles *with a ptxas advisory warning* and lowers to a plain **unicast** `UTMALDG.2D` — byte-identical to the non-multicast form, versus real multicast on B200 — so compile-acceptance here doesn't imply a hardware multicast fast-path. ([`probes/j_multicast_tma.cu`](probes/j_multicast_tma.cu).)
- **Stochastic rounding (`cvt.rs`) isn't even camp-wide.** It's **B200 and B300 only** — Thor rejects it with the same error as the 5090. So does `redux.sync.{min,max}.f32`. The persistent folklore that Spark (GB10) has `cvt.rs` is refuted by `ptxas` and by every PTX manual back to 8.8.
- **The 120f family target loses exactly two instructions vs the 5090** — the *sparse* block-scaled fp4 MMAs (`a`-suffix-exclusive). So dense nvfp4 kernels compile once for `sm_120f` and run on the 5090, the RTX PRO 6000, and Spark from one binary; the belief that the family target "can't do NVFP4 MMA" is true only for the sparse variant.
- **B300 vs B200: zero differences** across everything probed.

## The matrix

Legend: ✅ assembles · ❌ arch-reject · ⚠️ assembles, but read the footnote

| Instruction family | B200 (100a) | B300 (103a) | Thor (110a) | RTX 50 (120a) | 120f family | Spark (121a) |
|---|---|---|---|---|---|---|
| `mma.sync` f16/bf16/tf32/int (baselines) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mma.sync` fp8 (e4m3/e5m2, f32 & f16 acc) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mma.sync` **fp6/fp4** (`kind::f8f6f4`, all 21 mixes) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `mma.sync` block-scaled **nvfp4** (`kind::mxf4nvf4`, the A4Q instruction) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `mma.sync` block-scaled mxfp4 / mxfp8-fp6-fp4 (`kind::mxf4` / `mxf8f6f4`, 25 A×B mixes) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Sparse** block-scaled fp4 (`mma.sp::ordered_metadata.kind::mxf4nvf4` k128) | ❌ | ❌ | ❌ | ✅ | **❌** | ✅ |
| `tcgen05.*` (tensor memory, the datacenter MMA world) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `tcgen05.mma...block_scale` (datacenter's nvfp4 path) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `wgmma.*` (Hopper's pride) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ ¹ |
| `cvt` packed fp8/fp6/fp4 ↔ f32/f16x2/bf16x2 (the whole convert zoo) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cvt.rs.*` — **stochastic rounding** | ✅ | ✅ | **❌** | ❌ | ❌ | ❌ |
| `ldmatrix` m8n8 b16 (±trans), `stmatrix`, `movmatrix` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ldmatrix` sub-byte: `b4x16_p64` (fp4), `b6x16_p32` (fp6), m16n16 b8 trans | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cp.async.bulk` + full **TMA** (`bulk.tensor` 1d–5d, im2col, prefetch) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ² |
| TMA `.multicast::cluster` | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ ³ |
| `cp.async.bulk.tensor...cta_group::2` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `mbarrier.*` incl. `expect_tx` (the TMA sync machinery) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Clusters: `barrier.cluster`, `mapa`, `ld/st.shared::cluster`, DSM | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ⁴ |
| `redux.sync` (warp reduce, int) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `redux.sync.{min,max}.f32` | ✅ | ✅ | **❌** | ❌ | ❌ | ❌ |
| `multimem.ld_reduce` (NVLink/symm-mem loads) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ⁵ |
| `griddepcontrol.*` (programmatic dependent launch) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `setmaxnreg` | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ ⁶ |
| `u8x4`/`s8x4` SIMD int ops (PTX 9.2) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ ⁷ |
| 3-input min/max (`FMNMX3`/`VIMNMX3`, SASS at -O3) | ✅ | ✅ | ⚠️ | **❌** | **❌** | **❌** ⁸ |

¹ `wgmma` is `sm_90a`-only. Not on *any* Blackwell, either camp.

² Real `TMALDG`/`TMASTG` SASS on consumer, byte-identical machinery to B200 — for `shared::cta` destinations. Cluster-destination copies carry a compiler-inserted syscall guard on the remote-rank path.

³ Accepted on 12.x with an actual `ptxas` *advisory warning*, and it lowers to a plain **unicast** `UTMALDG.2D` — byte-identical to the non-multicast form, zero `UTMALDG.MULTICAST` opcodes, versus real multicast on B200. So on 12.x, compile-acceptance here doesn't imply a hardware multicast path — confirmed at the SASS level.

⁴ The entire cluster ISA assembles to genuine CGA hardware SASS on consumer (`CGABAR_ARV`/`CGABAR_WAIT`, same bytes as B200) — **and it launches at runtime.** On an RTX PRO 6000, multi-CTA clusters launch cleanly, distributed shared memory works (a rank reads another rank's smem through `cluster.map_shared_rank`, verified against a no-barrier race control), and the passing kernel disassembles to real `UCGABAR_ARV`/`UCGABAR_WAIT`. Portable cluster size 8 works across the consumer camp; `sm120` also takes 16 via `NonPortableClusterSizeAllowed`, while Spark (`sm121`) caps at 8.

⁵ Real `LDGMC` multicast-load SASS on the 5090, not emulation. `multimem.st`/`red` lower to plain stores on every arch including B200.

⁶ Assembles everywhere, compiles to zero SASS in a trivial kernel everywhere. Compile acceptance tells you nothing about this instruction.

⁷ Now probed. The PTX 9.2 SIMD byte ops — `add`/`sub`/`min`/`max`/`neg` on `.u8x4`/`.s8x4`, plus saturating `add.sat`/`sub.sat` — assemble on the **whole warp-block-scale camp** (`sm_120a`, `sm_120f`, `sm_121a`) and arch-reject on the entire tcgen05 camp (`sm_100a`, `sm_103a`, `sm_110a`). "120f" is the *family* label (a feature promoted to `sm_120f` is exposed across the whole 12.x family, exactly like the dense nvfp4 MMA), not an exclusion of the 120a/121a targets. On `sm_120a` each lowers to one real SIMD instruction — `VIADD.U8`, `VIMNMX.U8`/`.S8`, `VIADD.32.ISAT` — genuine byte-lane hardware. (There is no packed `.u8x4` abs-diff in either camp; only the legacy `vabsdiff4.u32`, which assembles everywhere.)

⁸ Not a PTX instruction — a SASS fusion (chained `fmax`/`max` at -O3), probed on all six targets plus a Hopper (`sm_90a`) control ([`probes/fmnmx3.cu`](probes/fmnmx3.cu)). It splits by **type**. *Float* `FMNMX3` fuses across the **entire tcgen05 camp** — B200, B300 **and Thor** — and on none of the consumer camp (120a/120f/121a emit two `FMNMX`). *Integer* `VIMNMX3` is narrower: Hopper, B200 and B300 have it, but **Thor does not** (falls back to `IMNMX` pairs), and consumer has neither (`VIMNMX` pairs). Net: B200/B300 = float+int, **Thor = float-only** (the mirror image of Hopper's int-only), consumer Blackwell = neither. The regression is a consumer-camp fact, byte-identical on 120a, 120f *and* 121a.

**Want the raw verdicts?** Every cell above is derived from [`results/results.json`](results/results.json) (296 probes) and [`results/gap_results.json`](results/gap_results.json) (39 gap-fill probes). Regenerate the accept/reject matrix and the camp summary straight from that data — nothing hand-typed — with:

```
python results/summarize.py
```

## How to run

**Requirements:** a CUDA 13.2 toolkit on `PATH` (`ptxas`, `nvcc`, `cuobjdump`, `nvdisasm`) and Python 3. The results in this repo are a **version snapshot**: `ptxas` **V13.2.78** for the main 296-probe sweep, **V13.2.51** for the gap-fill. No GPU is needed for the compile probes (`ptxas` is a cross-assembler); the cluster runtime probe ([`probes/j_cluster_probe.cu`](probes/j_cluster_probe.cu)) does need a consumer Blackwell card.

Main sweep — generate the probe manifest, then run the oracle:

```
python generators/gen_probes.py                     # -> generators/probes.json (258 core probes, families 1-7)
python harness/runner.py generators/probes.json     # -> results.json (ptxas verdict per target)
```

`runner.py` wraps each snippet in a minimal `.ptx` (`.version 8.8`, auto-upgraded to whatever `ptxas` demands), runs `ptxas -arch=<target>` for each arch in `ARCHES`, and records `ACCEPT` / `REJECT(class)`. Edit `ARCHES` (or pass per-probe `arches`) to pick targets. `harness/extend_archs.py` shows how the datacenter/Thor columns (`sm_103a`, `sm_110a`) were added to an existing `results.json`, including the PTX-version-floor fix for Thor (`sm_110a` needs PTX ISA ≥ 9.0).

Gap-fill (the two corrected rows) — the `u8x4`/`s8x4` SIMD sweep, the fp4↔fp8 convert probes, and the `FMNMX3` SASS check:

```
python generators/gen_gap_probes.py                 # u8x4/s8x4 SIMD int (footnote 7)
python generators/gen_conv_probes.py                # fp4<->fp8/fp6 packed converts
python generators/gen_sinks.py sm_120a              # load->op->store kernels for SASS confirmation
# harness/run_gap.sh is the full as-run orchestrator (assumes a flat working dir; see the note in the script)
```

**The arch-family caveat.** `sm_120f` is a *family* target: a feature promoted to it is exposed across the whole cc-12.x family (`sm_120a`, `sm_121a`, …), not withheld from them. So a `sm_120f`-accepted instruction being "120f" does **not** mean "120f-only" — it means "12.x-family-wide." The two `a`-suffix-exclusive features found (sparse block-scale fp4) are the exception that proves the rule. Guard predicates should encode *which* kind of limit they mean: dense block-scale MMA by **family** (`__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200`), sparse block-scale by `(120a || 121a)` exactly, `cvt.rs` by `(100a || 103a)` — not "datacenter," not Thor.

## Caveats

- **This is `ptxas` 13.2's worldview.** Older toolchains reject things this one accepts (there are community reports of 12.8-era `ptxas` refusing dense nvfp4 on 120f); newer ones will differ again. A snapshot with a version number, not scripture.
- **Acceptance ≠ performance.** Everything relied on was disassembled to confirm real hardware lowering — that's how the multicast syscall trampoline and the zero-SASS `setmaxnreg` got caught. For compile-only rows, ✅ means "the ISA admits it," nothing about throughput.
- **The compile oracle can't see the runtime.** For the cluster row — where the method's honest limit met a decade of "5090s can't launch clusters" folklore — we ran the 20-line runtime experiment. Docs right, folklore wrong.
- **Six targets, one harness bug we caught.** The first Thor column was garbage: `sm_110` needs PTX 9.0+ and the auto-version-upgrade didn't catch that error form, so 134 cells recorded "rejected" that were really "harness sent an old dialect." Re-probed and fixed (`extend_archs.py`) — a reminder that compile-probing lies most confidently about the target you added last.
- **296 probes is curated, not exhaustive** — the tensor/memory/sync surface a serving-kernel author cares about, not all of PTX.

## Repo layout

```
generators/
  gen_probes.py        the core generator: 258 single-instruction probes, families 1-7
  gen_gap_probes.py    u8x4/s8x4 SIMD int gap-fill (footnote 7)
  gen_conv_probes.py   fp4<->fp8/fp6 packed-convert probes
  gen_sinks.py         load->op->store kernels for SASS confirmation
harness/
  runner.py            the ptxas oracle: wrap one instruction, assemble per target, record the verdict
  extend_archs.py      add sm_103a/sm_110a columns (+ the Thor PTX-version-floor fix)
  run_gap.sh           the gap-fill orchestrator (as-run record; flat-dir layout)
probes/
  fmnmx3.cu            3-input min/max SASS fusion probe (FMNMX3 / VIMNMX3)
  j_cluster_probe.cu   the cluster RUNTIME experiment (launch + DSM + cluster.sync)
  j_multicast_tma.cu   multicast-TMA SASS check (unicast-lowering confirmation)
results/
  results.json         the full 296-probe x 6-target verdict matrix (the value)
  gap_results.json     the 39 gap-fill probes (SIMD int + converts), 6 targets
  summarize.py         regenerate the accept/reject matrix + camp summary from the JSON
```

The full 296-probe matrix ([`results/results.json`](results/results.json)) also carries families `8_extra` and `9_sink` (SASS-dump variants and a handful of extra ops) that the additional probe scripts emitted during the campaign; they are preserved in the data even though the two small generators for them are not shipped here. `gen_probes.py` reproduces the 258-probe core.

## Provenance

The probes, the harness, the six-target sweeps, the SASS verification, and this writeup were generated by **Claude** under a charter set by **Jetha Chan**, who owns the framing and the sign-off. The matrix is machine-generated and machine-checked — the raw `results.json` ships with it, and [`results/summarize.py`](results/summarize.py) regenerates every claim in the table from that data. If a cell here disagrees with your toolchain, the JSON and the generator are right there to re-run.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Jetha Chan.
