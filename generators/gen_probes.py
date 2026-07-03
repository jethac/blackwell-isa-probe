#!/usr/bin/env python3
"""Generate probes.json - the full probe matrix."""
import json, os

P = []
def add(family, name, code, **kw):
    d = {"family": family, "name": name, "code": code}
    d.update(kw)
    P.append(d)

D4 = "{f0,f1,f2,f3}"   # f32 d/c
A4 = "{a0,a1,a2,a3}"
B2 = "{b0,b1}"
B4 = "{b0,b1,b2,b3}"
SF = "{sfa},{0,0},{sfb},{0,0}"

# ---------------- Family 1: block-scaled MMA ----------------
# kind x shape x scale_vec x sf-type, e2m1xe2m1
for kind in ["mxf4", "mxf4nvf4"]:
    for shape in ["m16n8k64", "m16n8k32"]:
        for sv in ["1X", "2X", "4X"]:
            for sf in ["ue8m0", "ue4m3"]:
                b = B2 if shape.endswith("k64") else "{b0}"
                add("1_blockscale_mma",
                    f"{kind}.{shape}.sv{sv}.{sf}.e2m1xe2m1",
                    f"mma.sync.aligned.kind::{kind}.block_scale.scale_vec::{sv}.{shape}.row.col"
                    f".f32.e2m1.e2m1.f32.{sf} {D4},{A4},{b},{D4},{SF};")
# mxf8f6f4: full 5x5 dtype cross at k32, 1X ue8m0
FTS = ["e4m3", "e5m2", "e3m2", "e2m3", "e2m1"]
for da in FTS:
    for db in FTS:
        add("1_blockscale_mma", f"mxf8f6f4.k32.sv1X.ue8m0.{da}x{db}",
            f"mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col"
            f".f32.{da}.{db}.f32.ue8m0 {D4},{A4},{B2},{D4},{SF};")
# mxf8f6f4 off-spec: ue4m3 sf, 2X/4X, k64
add("1_blockscale_mma", "mxf8f6f4.k32.sv1X.ue4m3.e4m3xe4m3",
    f"mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue4m3 {D4},{A4},{B2},{D4},{SF};")
add("1_blockscale_mma", "mxf8f6f4.k32.sv2X.ue8m0.e4m3xe4m3",
    f"mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::2X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {D4},{A4},{B2},{D4},{SF};")
add("1_blockscale_mma", "mxf8f6f4.k64.sv1X.ue8m0.e4m3xe4m3",
    f"mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k64.row.col.f32.e4m3.e4m3.f32.ue8m0 {D4},{A4},{B4},{D4},{SF};")
# mixed f4 kind dtypes (should reject: mxf4 kinds are e2m1 only)
add("1_blockscale_mma", "mxf4nvf4.k64.sv4X.ue4m3.e4m3xe2m1",
    f"mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e4m3.e2m1.f32.ue4m3 {D4},{A4},{B2},{D4},{SF};")
# f16 accumulator with block_scale (expect reject)
add("1_blockscale_mma", "mxf4nvf4.k64.sv4X.ue4m3.f16acc",
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col"
    ".f16.e2m1.e2m1.f16.ue4m3 {hx0,hx1},{a0,a1,a2,a3},{b0,b1},{hx0,hx1}," + SF + ";")
# col.row layout (expect reject; only row.col)
add("1_blockscale_mma", "mxf4nvf4.k64.sv4X.ue4m3.colrow",
    f"mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.col.row.f32.e2m1.e2m1.f32.ue4m3 {D4},{A4},{B2},{D4},{SF};")

# ---------------- Family 2: plain / sparse mma ----------------
add("2_plain_mma", "m16n8k16.f32.f16.f16.f32",
    f"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {D4},{{a0,a1,a2,a3}},{B2},{D4};")
add("2_plain_mma", "m16n8k16.f32.bf16.bf16.f32",
    f"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {D4},{A4},{B2},{D4};")
add("2_plain_mma", "m16n8k8.f32.tf32.tf32.f32",
    f"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {D4},{A4},{B2},{D4};")
# legacy (sm_89-style, no kind::) fp8
for da in ["e4m3", "e5m2"]:
    for db in ["e4m3", "e5m2"]:
        add("2_plain_mma", f"legacy.k32.f32.{da}.{db}.f32",
            f"mma.sync.aligned.m16n8k32.row.col.f32.{da}.{db}.f32 {D4},{A4},{B2},{D4};")
# kind::f8f6f4 full cross, f32 acc
for da in FTS:
    for db in FTS:
        add("2_plain_mma", f"kindf8f6f4.k32.f32.{da}.{db}.f32",
            f"mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.{da}.{db}.f32 {D4},{A4},{B2},{D4};")
# f16 accumulator fp8
add("2_plain_mma", "legacy.k32.f16.e4m3.e4m3.f16",
    "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {hx0,hx1},{a0,a1,a2,a3},{b0,b1},{hx0,hx1};")
add("2_plain_mma", "kindf8f6f4.k32.f16.e4m3.e4m3.f16",
    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e4m3.f16 {hx0,hx1},{a0,a1,a2,a3},{b0,b1},{hx0,hx1};")
# int8 / s4 baselines
add("2_plain_mma", "m16n8k32.s32.s8.s8.s32",
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {r0,r1,r2,r3},{a0,a1,a2,a3},{b0,b1},{r0,r1,r2,r3};")
add("2_plain_mma", "m16n8k64.s32.s4.s4.s32",
    "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 {r0,r1,r2,r3},{a0,a1,a2,a3},{b0,b1},{r0,r1,r2,r3};")
# e2m1 without kind (expect reject)
add("2_plain_mma", "nokindf4.k32.f32.e2m1.e2m1.f32",
    f"mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32 {D4},{A4},{B2},{D4};")
# sparse mma
add("2_sparse_mma", "sp.m16n8k32.f32.f16.f16.f32",
    f"mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {D4},{A4},{B4},{D4},e0,0x0;")
add("2_sparse_mma", "sp_om.m16n8k32.f32.f16.f16.f32",
    f"mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {D4},{A4},{B4},{D4},e0,0x0;")
add("2_sparse_mma", "sp.m16n8k64.f32.e4m3.e4m3.f32",
    f"mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 {D4},{A4},{B4},{D4},e0,0x0;")
add("2_sparse_mma", "sp_om.m16n8k64.f32.e4m3.e4m3.f32",
    f"mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 {D4},{A4},{B4},{D4},e0,0x0;")
add("2_sparse_mma", "sp_om.kindf8f6f4.m16n8k64.f32.e4m3.e4m3.f32",
    f"mma.sp::ordered_metadata.sync.aligned.kind::f8f6f4.m16n8k64.row.col.f32.e4m3.e4m3.f32 {D4},{A4},{B4},{D4},e0,0x0;")
add("2_sparse_mma", "sp_om.kindf8f6f4.m16n8k64.f32.e2m1.e2m1.f32",
    f"mma.sp::ordered_metadata.sync.aligned.kind::f8f6f4.m16n8k64.row.col.f32.e2m1.e2m1.f32 {D4},{A4},{B4},{D4},e0,0x0;")
add("2_sparse_mma", "sp_om.mxf4nvf4.bs.k128.sv4X.ue4m3",
    f"mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 {D4},{A4},{B4},{D4},e0,0x0,{SF};")
add("2_sparse_mma", "sp_om.mxf4.bs.k128.sv2X.ue8m0",
    f"mma.sp::ordered_metadata.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0 {D4},{A4},{B4},{D4},e0,0x0,{SF};")
add("2_sparse_mma", "sp_om.mxf8f6f4.bs.k64.sv1X.ue8m0",
    f"mma.sp::ordered_metadata.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k64.row.col.f32.e4m3.e4m3.f32.ue8m0 {D4},{A4},{B4},{D4},e0,0x0,{SF};")

# ---------------- Family 3: cvt ----------------
# pack: f32x2 -> fp8/fp6/fp4/ue8m0 x2
pk = [("e2m1x2", "q0"), ("e2m3x2", "h0"), ("e3m2x2", "h0"),
      ("e4m3x2", "h0"), ("e5m2x2", "h0"), ("ue8m0x2", "h0")]
for t, dreg in pk:
    for rnd in ["rn", "rz", "rp"]:
        add("3_cvt", f"{t}.f32.{rnd}.satfinite",
            f"cvt.{rnd}.satfinite.{t}.f32 {dreg}, f0, f1;")
    add("3_cvt", f"{t}.f32.rn.nosat", f"cvt.rn.{t}.f32 {dreg}, f0, f1;")
# unpack: x2 -> f16x2 / bf16x2
for t, sreg in pk:
    add("3_cvt", f"f16x2.{t}", f"cvt.rn.f16x2.{t} hx0, {sreg};")
    add("3_cvt", f"bf16x2.{t}", f"cvt.rn.bf16x2.{t} r5, {sreg};")
# from f16x2
for t in ["e4m3x2", "e5m2x2", "e2m1x2", "e3m2x2", "e2m3x2"]:
    dreg = "q0" if t == "e2m1x2" else "h0"
    add("3_cvt", f"{t}.f16x2.rn.satfinite", f"cvt.rn.satfinite.{t}.f16x2 {dreg}, hx0;")
add("3_cvt", "ue8m0x2.bf16x2.rz.satfinite", "cvt.rz.satfinite.ue8m0x2.bf16x2 h0, r5;")
# stochastic rounding .rs (x4 packs, rbits operand)
add("3_cvt", "rs.f16x2.f32", "cvt.rs.f16x2.f32 hx0, f0, f1, r0;")
add("3_cvt", "rs.bf16x2.f32", "cvt.rs.bf16x2.f32 r5, f0, f1, r0;")
add("3_cvt", "rs.satfinite.e4m3x4.f32", "cvt.rs.satfinite.e4m3x4.f32 r1, {f0,f1,f2,f3}, r0;")
add("3_cvt", "rs.satfinite.e5m2x4.f32", "cvt.rs.satfinite.e5m2x4.f32 r1, {f0,f1,f2,f3}, r0;")
add("3_cvt", "rs.satfinite.e2m1x4.f32", "cvt.rs.satfinite.e2m1x4.f32 h0, {f0,f1,f2,f3}, r0;")
add("3_cvt", "rs.satfinite.e3m2x4.f32", "cvt.rs.satfinite.e3m2x4.f32 r1, {f0,f1,f2,f3}, r0;")
add("3_cvt", "rs.satfinite.e2m3x4.f32", "cvt.rs.satfinite.e2m3x4.f32 r1, {f0,f1,f2,f3}, r0;")
add("3_cvt", "rs.ue8m0x4.f32", "cvt.rs.ue8m0x4.f32 r1, {f0,f1,f2,f3}, r0;")
# baselines
add("3_cvt", "rn.f16x2.f32", "cvt.rn.f16x2.f32 hx0, f0, f1;")
add("3_cvt", "rn.bf16x2.f32", "cvt.rn.bf16x2.f32 r5, f0, f1;")
add("3_cvt", "rn.satfinite.e4m3x2.f32.relu", "cvt.rn.satfinite.relu.e4m3x2.f32 h0, f0, f1;")

# ---------------- Family 4: matrix ld/st ----------------
mdecl = ".shared .align 128 .b8 sbuf[2048];"
for num, regs in [("x1", "{r0}"), ("x2", "{r0,r1}"), ("x4", "{r0,r1,r2,r3}")]:
    add("4_ldstmatrix", f"ldmatrix.m8n8.{num}.b16",
        f"ldmatrix.sync.aligned.m8n8.{num}.shared.b16 {regs}, [rd0];", decls=mdecl)
    add("4_ldstmatrix", f"ldmatrix.m8n8.{num}.trans.b16",
        f"ldmatrix.sync.aligned.m8n8.{num}.trans.shared.b16 {regs}, [rd0];", decls=mdecl)
    add("4_ldstmatrix", f"stmatrix.m8n8.{num}.b16",
        f"stmatrix.sync.aligned.m8n8.{num}.shared.b16 [rd0], {regs};", decls=mdecl)
# m16n16 b8 loads (require .trans)
add("4_ldstmatrix", "ldmatrix.m16n16.x1.trans.b8",
    "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {r0,r1}, [rd0];", decls=mdecl)
add("4_ldstmatrix", "ldmatrix.m16n16.x2.trans.b8",
    "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {r0,r1,r2,r3}, [rd0];", decls=mdecl)
# m8n16 b8 source-format loads (fp6/fp4 padded)
for num, regs in [("x1", "{r0}"), ("x2", "{r0,r1}"), ("x4", "{r0,r1,r2,r3}")]:
    add("4_ldstmatrix", f"ldmatrix.m8n16.{num}.b8x16.b6x16_p32",
        f"ldmatrix.sync.aligned.m8n16.{num}.shared.b8x16.b6x16_p32 {regs}, [rd0];", decls=mdecl)
    add("4_ldstmatrix", f"ldmatrix.m8n16.{num}.b8x16.b4x16_p64",
        f"ldmatrix.sync.aligned.m8n16.{num}.shared.b8x16.b4x16_p64 {regs}, [rd0];", decls=mdecl)
# stmatrix m16n8 b8
add("4_ldstmatrix", "stmatrix.m16n8.x1.trans.b8",
    "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [rd0], {r0};", decls=mdecl)
add("4_ldstmatrix", "movmatrix.m8n8.trans.b16",
    "movmatrix.sync.aligned.m8n8.trans.b16 r0, r1;")

# ---------------- Family 5: async data movement ----------------
sdecl = ".shared .align 16 .b8 sbuf[1024];\n.shared .align 8 .b64 mbar;"
add("5_async", "cp.async.ca.shared.global",
    "cp.async.ca.shared.global [rd0], [rd1], 16;", decls=sdecl)
add("5_async", "cp.async.bulk.s2g",
    "cp.async.bulk.global.shared::cta.bulk_group [rd0], [rd1], r0;", decls=sdecl)
add("5_async", "cp.async.bulk.g2s.cluster",
    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [rd0], [rd1], r0, [rd2];", decls=sdecl)
add("5_async", "cp.async.bulk.g2s.cta",
    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [rd0], [rd1], r0, [rd2];", decls=sdecl)
add("5_async", "cp.async.bulk.s2s.cluster",
    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [rd0], [rd1], r0, [rd2];", decls=sdecl)
add("5_async", "cp.async.bulk.prefetch.L2",
    "cp.async.bulk.prefetch.L2.global [rd1], r0;", decls=sdecl)
for d, idx in [("1d", "{r0}"), ("2d", "{r0,r1}"), ("3d", "{r0,r1,r2}"),
               ("4d", "{r0,r1,r2,r3}"), ("5d", "{r0,r1,r2,r3,r4}")]:
    add("5_async", f"cp.async.bulk.tensor.{d}.g2s.cluster",
        f"cp.async.bulk.tensor.{d}.shared::cluster.global.mbarrier::complete_tx::bytes [rd0], [rd1, {idx}], [rd2];", decls=sdecl)
    add("5_async", f"cp.async.bulk.tensor.{d}.g2s.cta",
        f"cp.async.bulk.tensor.{d}.shared::cta.global.mbarrier::complete_tx::bytes [rd0], [rd1, {idx}], [rd2];", decls=sdecl)
    add("5_async", f"cp.async.bulk.tensor.{d}.s2g",
        f"cp.async.bulk.tensor.{d}.global.shared::cta.bulk_group [rd1, {idx}], [rd0];", decls=sdecl)
add("5_async", "cp.async.bulk.tensor.3d.im2col.g2s.cluster",
    "cp.async.bulk.tensor.3d.im2col.shared::cluster.global.mbarrier::complete_tx::bytes [rd0], [rd1, {r0,r1,r2}], [rd2], {h0};", decls=sdecl)
add("5_async", "cp.async.bulk.tensor.4d.im2col.g2s.cta",
    "cp.async.bulk.tensor.4d.im2col.shared::cta.global.mbarrier::complete_tx::bytes [rd0], [rd1, {r0,r1,r2,r3}], [rd2], {h0,h1};", decls=sdecl)
add("5_async", "cp.async.bulk.prefetch.tensor.2d.L2",
    "cp.async.bulk.prefetch.tensor.2d.L2.global [rd1, {r0,r1}];", decls=sdecl)
add("5_async", "prefetch.tensormap", "prefetch.tensormap [rd1];", decls=sdecl)
add("5_async", "cp.reduce.async.bulk.add.f32",
    "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [rd0], [rd1], r0;", decls=sdecl)
add("5_async", "cp.reduce.async.bulk.add.u32",
    "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [rd0], [rd1], r0;", decls=sdecl)
add("5_async", "st.async.cluster.mbar.b32",
    "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [rd0], r0, [rd1];", decls=sdecl)
add("5_async", "red.async.cluster.mbar.add.u32",
    "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32 [rd0], r0, [rd1];", decls=sdecl)
add("5_async", "mbarrier.init.b64", "mbarrier.init.shared::cta.b64 [rd0], 32;", decls=sdecl)
add("5_async", "mbarrier.arrive.b64", "mbarrier.arrive.shared::cta.b64 rd1, [rd0];", decls=sdecl)
add("5_async", "mbarrier.arrive.expect_tx.b64",
    "mbarrier.arrive.expect_tx.shared::cta.b64 rd1, [rd0], r0;", decls=sdecl)
add("5_async", "mbarrier.arrive.expect_tx.cluster.b64",
    "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [rd0], r0;", decls=sdecl)
add("5_async", "mbarrier.expect_tx.b64", "mbarrier.expect_tx.shared::cta.b64 [rd0], r0;", decls=sdecl)
add("5_async", "mbarrier.complete_tx.b64", "mbarrier.complete_tx.shared::cta.b64 [rd0], r0;", decls=sdecl)
add("5_async", "mbarrier.try_wait.parity.b64",
    "mbarrier.try_wait.parity.shared::cta.b64 p0, [rd0], r0;", decls=sdecl)
add("5_async", "mbarrier.test_wait.b64",
    "mbarrier.test_wait.shared::cta.b64 p0, [rd0], rd1;", decls=sdecl)
add("5_async", "fence.proxy.async", "fence.proxy.async;")
add("5_async", "fence.proxy.async.shared_cta", "fence.proxy.async.shared::cta;")
add("5_async", "fence.proxy.async.global", "fence.proxy.async.global;")
add("5_async", "fence.proxy.alias", "fence.proxy.alias;")
add("5_async", "cp.async.bulk.commit_group", "cp.async.bulk.commit_group;")
add("5_async", "cp.async.bulk.wait_group", "cp.async.bulk.wait_group 0;")
add("5_async", "cp.async.bulk.wait_group.read", "cp.async.bulk.wait_group.read 0;")
add("5_async", "tensormap.replace.global_address",
    "tensormap.replace.tile.global_address.global.b1024.b64 [rd0], rd1;")
add("5_async", "tensormap.cp_fenceproxy",
    "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [rd0], [rd1], 128;", decls=sdecl)
add("5_async", "st.bulk.weak", "st.bulk.weak.shared::cta [rd0], rd1, 0;", decls=sdecl)

# ---------------- Family 6: cluster ----------------
cdecl = ".shared .align 16 .b8 sbuf[1024];"
add("6_cluster", "barrier.cluster.arrive", "barrier.cluster.arrive;")
add("6_cluster", "barrier.cluster.wait", "barrier.cluster.wait;")
add("6_cluster", "barrier.cluster.arrive.relaxed", "barrier.cluster.arrive.relaxed;")
add("6_cluster", "mapa.shared_cluster.u32",
    "mapa.shared::cluster.u32 r0, r1, r2;", decls=cdecl)
add("6_cluster", "mapa.u64", "mapa.u64 rd0, rd1, r2;")
add("6_cluster", "getctarank.shared_cluster.u32",
    "getctarank.shared::cluster.u32 r0, r1;", decls=cdecl)
add("6_cluster", "ld.shared_cluster.u32", "ld.shared::cluster.u32 r0, [rd0];", decls=cdecl)
add("6_cluster", "st.shared_cluster.u32", "st.shared::cluster.u32 [rd0], r0;", decls=cdecl)
add("6_cluster", "atom.shared_cluster.add.u32",
    "atom.shared::cluster.add.u32 r0, [rd0], r1;", decls=cdecl)
add("6_cluster", "sreg.cluster_ctarank", "mov.u32 r0, %cluster_ctarank;")
add("6_cluster", "sreg.clusterid", "mov.u32 r0, %clusterid.x;")
add("6_cluster", "sreg.is_explicit_cluster",
    "mov.pred p0, %is_explicit_cluster;")
CLUSTER_RAW = """\
.version {version}
.target {target}
.address_size 64

.visible .entry probe() .explicitcluster .reqnctapercluster 2, 1, 1
{{
    barrier.cluster.arrive;
    barrier.cluster.wait;
    ret;
}}
"""
add("6_cluster", "entry.explicitcluster.reqnctapercluster", "", raw=CLUSTER_RAW)
REQNCTAPERSM_RAW = """\
.version {version}
.target {target}
.address_size 64

.visible .entry probe() .maxnctapersm 2
{{
    ret;
}}
"""
add("6_cluster", "entry.maxnctapersm", "", raw=REQNCTAPERSM_RAW)

# ---------------- Family 7: misc ----------------
add("7_misc", "redux.sync.add.s32", "redux.sync.add.s32 r0, r1, 0xffffffff;")
add("7_misc", "redux.sync.max.u32", "redux.sync.max.u32 r0, r1, 0xffffffff;")
add("7_misc", "redux.sync.min.s32", "redux.sync.min.s32 r0, r1, 0xffffffff;")
add("7_misc", "redux.sync.and.b32", "redux.sync.and.b32 r0, r1, 0xffffffff;")
add("7_misc", "redux.sync.min.f32", "redux.sync.min.f32 f0, f1, 0xffffffff;")
add("7_misc", "redux.sync.max.f32", "redux.sync.max.f32 f0, f1, 0xffffffff;")
add("7_misc", "redux.sync.max.abs.f32", "redux.sync.max.abs.f32 f0, f1, 0xffffffff;")
add("7_misc", "redux.sync.max.NaN.f32", "redux.sync.max.NaN.f32 f0, f1, 0xffffffff;")
add("7_misc", "redux.sync.add.f32", "redux.sync.add.f32 f0, f1, 0xffffffff;")
add("7_misc", "elect.sync", "elect.sync r0|p0, 0xffffffff;")
add("7_misc", "griddepcontrol.launch_dependents", "griddepcontrol.launch_dependents;")
add("7_misc", "griddepcontrol.wait", "griddepcontrol.wait;")
add("7_misc", "multimem.ld_reduce.add.f32", "multimem.ld_reduce.global.add.f32 f0, [rd0];")
add("7_misc", "multimem.st.f32", "multimem.st.global.f32 [rd0], f0;")
add("7_misc", "multimem.red.add.f32", "multimem.red.global.add.f32 [rd0], f0;")
tdecl = ".shared .align 16 .b8 sbuf[1024];"
add("7_misc", "tcgen05.alloc",
    "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [rd0], 32;", decls=tdecl)
add("7_misc", "tcgen05.ld", "tcgen05.ld.sync.aligned.32x32b.x1.b32 {r0}, [r1];")
add("7_misc", "tcgen05.commit",
    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [rd0];", decls=tdecl)
add("7_misc", "wgmma.fence", "wgmma.fence.sync.aligned;")
add("7_misc", "wgmma.commit_group", "wgmma.commit_group.sync.aligned;")
add("7_misc", "wgmma.mma_async.m64n8k16.f16",
    "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {f0,f1,f2,f3}, rd0, rd1, p0, 1, 1, 0, 0;",
    arches=["sm_120a", "sm_121a", "sm_120f", "sm_100a", "sm_90a"])
add("7_misc", "setmaxnreg.inc", "setmaxnreg.inc.sync.aligned.u32 96;",
    arches=["sm_120a", "sm_121a", "sm_120f", "sm_100a", "sm_90a"])
add("7_misc", "setmaxnreg.dec", "setmaxnreg.dec.sync.aligned.u32 64;")
add("7_misc", "clusterlaunchcontrol.try_cancel",
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [rd0], [rd1];", decls=tdecl)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes.json")
with open(out, "w") as f:
    json.dump(P, f, indent=1)
print(f"{len(P)} probes written to {out}")
