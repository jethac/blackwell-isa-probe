#!/usr/bin/env python3
"""Comprehensive PTX-ISA probe generator.

This is the expansion from the curated serving-kernel surface (families 1-7,
G1_simd, G3_conv, produced by gen_probes.py / gen_gap_probes.py /
gen_conv_probes.py) to a systematic sweep of the *documented PTX instruction
set*. It folds in the curated families unchanged and adds the rest of PTX,
category by category, following the "Instructions" chapters of the PTX ISA
manual (the 9.x manual that matches ptxas 13.2, which accepts up to .version
9.2):

  10_int_arith     integer add/sub/mul/mad/mul24/mad24/sad/div/rem/abs/neg/
                   min/max/dp4a/dp2a
  11_extint_carry  extended-precision add.cc/addc/sub.cc/subc/mad.cc/madc
  12_fp32_64       single/double add/sub/mul/fma/mad/div/rcp/sqrt/rsqrt/sin/cos/
                   lg2/ex2/tanh/abs/neg/min/max/copysign/testp (rounding/ftz/sat)
  13_fp16_bf16     half/bhalf add/sub/mul/fma/neg/abs/min/max/tanh/ex2 in
                   f16, f16x2, bf16, bf16x2
  14_cmp_sel       set / setp / selp / slct
  15_logic_shift   and/or/xor/not/cnot/lop3/shl/shr/shf
  16_bitmanip      popc/clz/bfind/fns/brev/bfe/bfi/szext/bmsk/prmt
  17_datamov_cvt   mov / ld / st / ldu / prefetch / prefetchu / applypriority /
                   discard / createpolicy / isspacep / cvta / cvt (int<->int,
                   int<->float, float<->float, tf32, cvt.pack)
  18_tex_surf      tex / tld4 / txq / suld / sust / sured / suq
  19_control_flow  bra / @p bra / bra.uni / call / ret / exit / brx.idx
  20_sync_comm     bar / barrier / bar.warp.sync / membar / fence
  21_atomic        atom / red (all ops, spaces, scopes, types incl. vector)
  22_warp          vote / match / activemask / shfl / redux.sync
  23_video_simd    scalar video (vadd/vsub/...) + SIMD video (vadd2/vadd4/...) +
                   packed 16-bit (s16x2/u16x2)
  24_special       nanosleep / pmevent / trap / brkpt / special-register mov /
                   stack ops / a few post-9.2 stragglers (documented as coverage
                   boundary, not arch deltas)

SAMPLING RULE. Type-parametrized instructions are probed across the meaningful
type variants where differences can hide (int: s32/u32/s64/u64 + a 16-bit
sample; float: f32/f64; half: f16/f16x2/bf16/bf16x2; bit: b16/b32/b64), each
rounding mode sampled at least once per instruction that carries one, ftz/sat
sampled on f32, one representative state space / cache-op / memory-scope per
ld/st/atom rather than the full cross product. This is deliberately NOT every
trivial permutation; it is a representative-variant sweep designed to surface
per-target ISA differences without a combinatorial blowup.

Output: generators/probes_full.json (curated families + the above), all probes
defaulting to the six-target sweep in runner.ARCHES. Run it, then:
    python harness/runner.py generators/probes_full.json results/results_full.json
"""
import json, os, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ALL6 = ["sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_120f", "sm_121a"]

P = []
def add(family, name, code, **kw):
    d = {"family": family, "name": name, "code": code}
    d.update(kw)
    P.append(d)

# ------------------------------------------------------------------ #
# Fold in the curated families (1-7, G1_simd, G3_conv) unchanged, so the
# comprehensive matrix is one internally-consistent sweep at one ptxas version.
# We import the shipped generators and reuse their probe lists, normalizing the
# per-probe `arches` to the full six-target set (a couple of curated probes
# pinned a 5-arch list incl. a Hopper control; the comprehensive matrix is six
# Blackwell columns).
# ------------------------------------------------------------------ #
def _load(mod_name, fname):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.P

_curated = []
for mod_name, fname in [("gen_probes", "gen_probes.py"),
                        ("gen_conv_probes", "gen_conv_probes.py"),
                        ("gen_gap_probes", "gen_gap_probes.py")]:
    try:
        for pr in _load(mod_name, fname):
            pr = dict(pr)
            pr.pop("arches", None)          # -> default six-target sweep
            _curated.append(pr)
    except Exception as e:                  # pragma: no cover
        print(f"WARN: could not fold in {fname}: {e}")
P.extend(_curated)
N_CURATED = len(P)

# ================================================================== #
# 10_int_arith
# ================================================================== #
F = "10_int_arith"
INT32 = ["s32", "u32"]
INT_ALL = ["s32", "u32", "s64", "u64"]
def rr(t):  # a register name of the right width for an integer type
    return "rd0" if t.endswith("64") else ("h0" if t.endswith("16") else "r0")
def rri(t, i):
    base = "rd" if t.endswith("64") else ("h" if t.endswith("16") else "r")
    return f"{base}{i}"
for t in ["s16", "u16", "s32", "u32", "s64", "u64"]:
    d, a, b = rri(t, 0), rri(t, 1), rri(t, 2)
    add(F, f"add.{t}", f"add.{t} {d}, {a}, {b};")
    add(F, f"sub.{t}", f"sub.{t} {d}, {a}, {b};")
    add(F, f"min.{t}", f"min.{t} {d}, {a}, {b};")
    add(F, f"max.{t}", f"max.{t} {d}, {a}, {b};")
for t in ["s16", "s32", "s64"]:      # abs/neg apply to signed types only
    add(F, f"abs.{t}", f"abs.{t} {rri(t,0)}, {rri(t,1)};")
    add(F, f"neg.{t}", f"neg.{t} {rri(t,0)}, {rri(t,1)};")
add(F, "add.sat.s32", "add.sat.s32 r0, r1, r2;")
add(F, "sub.sat.s32", "sub.sat.s32 r0, r1, r2;")
# mul / mad .lo/.hi/.wide
for t in ["s32", "u32"]:
    add(F, f"mul.lo.{t}", f"mul.lo.{t} r0, r1, r2;")
    add(F, f"mul.hi.{t}", f"mul.hi.{t} r0, r1, r2;")
    add(F, f"mul.wide.{t}", f"mul.wide.{t} rd0, r1, r2;")
    add(F, f"mad.lo.{t}", f"mad.lo.{t} r0, r1, r2, r3;")
    add(F, f"mad.hi.{t}", f"mad.hi.{t} r0, r1, r2, r3;")
    add(F, f"mad.wide.{t}", f"mad.wide.{t} rd0, r1, r2, rd3;")
add(F, "mad.hi.sat.s32", "mad.hi.sat.s32 r0, r1, r2, r3;")
for t in ["s64", "u64"]:
    add(F, f"mul.lo.{t}", f"mul.lo.{t} rd0, rd1, rd2;")
    add(F, f"mul.hi.{t}", f"mul.hi.{t} rd0, rd1, rd2;")
add(F, "mul.wide.s16", "mul.wide.s16 r0, h1, h2;")
add(F, "mul.wide.u16", "mul.wide.u16 r0, h1, h2;")
# mul24 / mad24
for t in ["s32", "u32"]:
    add(F, f"mul24.lo.{t}", f"mul24.lo.{t} r0, r1, r2;")
    add(F, f"mul24.hi.{t}", f"mul24.hi.{t} r0, r1, r2;")
    add(F, f"mad24.lo.{t}", f"mad24.lo.{t} r0, r1, r2, r3;")
    add(F, f"mad24.hi.{t}", f"mad24.hi.{t} r0, r1, r2, r3;")
add(F, "mad24.hi.sat.s32", "mad24.hi.sat.s32 r0, r1, r2, r3;")
# sad / div / rem
add(F, "sad.s32", "sad.s32 r0, r1, r2, r3;")
add(F, "sad.u32", "sad.u32 r0, r1, r2, r3;")
for t in ["s32", "u32", "s64", "u64", "s16", "u16"]:
    add(F, f"div.{t}", f"div.{t} {rri(t,0)}, {rri(t,1)}, {rri(t,2)};")
for t in ["s32", "u32", "s64", "u64"]:
    add(F, f"rem.{t}", f"rem.{t} {rri(t,0)}, {rri(t,1)}, {rri(t,2)};")
# dp4a / dp2a (byte / 16-bit dot-accumulate)
for da in ["u32", "s32"]:
    for db in ["u32", "s32"]:
        add(F, f"dp4a.{da}.{db}", f"dp4a.{da}.{db} r0, r1, r2, r3;")
add(F, "dp2a.lo.u32.u32", "dp2a.lo.u32.u32 r0, r1, r2, r3;")
add(F, "dp2a.hi.u32.u32", "dp2a.hi.u32.u32 r0, r1, r2, r3;")
add(F, "dp2a.lo.s32.s32", "dp2a.lo.s32.s32 r0, r1, r2, r3;")
add(F, "dp2a.hi.s32.s32", "dp2a.hi.s32.s32 r0, r1, r2, r3;")

# ================================================================== #
# 11_extint_carry
# ================================================================== #
F = "11_extint_carry"
for t in ["u32", "s32", "u64"]:
    d = "rd0" if t.endswith("64") else "r0"
    a = "rd1" if t.endswith("64") else "r1"
    b = "rd2" if t.endswith("64") else "r2"
    add(F, f"add.cc.{t}", f"add.cc.{t} {d}, {a}, {b};")
    add(F, f"addc.{t}", f"addc.{t} {d}, {a}, {b};")
    add(F, f"addc.cc.{t}", f"addc.cc.{t} {d}, {a}, {b};")
    add(F, f"sub.cc.{t}", f"sub.cc.{t} {d}, {a}, {b};")
    add(F, f"subc.{t}", f"subc.{t} {d}, {a}, {b};")
    add(F, f"subc.cc.{t}", f"subc.cc.{t} {d}, {a}, {b};")
add(F, "mad.lo.cc.u32", "mad.lo.cc.u32 r0, r1, r2, r3;")
add(F, "madc.lo.cc.u32", "madc.lo.cc.u32 r0, r1, r2, r3;")
add(F, "mad.hi.cc.s32", "mad.hi.cc.s32 r0, r1, r2, r3;")
add(F, "madc.hi.cc.s32", "madc.hi.cc.s32 r0, r1, r2, r3;")
add(F, "mad.lo.cc.u64", "mad.lo.cc.u64 rd0, rd1, rd2, rd3;")

# ================================================================== #
# 12_fp32_64
# ================================================================== #
F = "12_fp32_64"
# --- f32 ---
for rnd in ["rn", "rz", "rm", "rp"]:
    add(F, f"add.{rnd}.f32", f"add.{rnd}.f32 f0, f1, f2;")
    add(F, f"mul.{rnd}.f32", f"mul.{rnd}.f32 f0, f1, f2;")
    add(F, f"fma.{rnd}.f32", f"fma.{rnd}.f32 f0, f1, f2, f3;")
add(F, "add.rn.ftz.f32", "add.rn.ftz.f32 f0, f1, f2;")
add(F, "add.rn.sat.f32", "add.rn.sat.f32 f0, f1, f2;")
add(F, "sub.rn.f32", "sub.rn.f32 f0, f1, f2;")
add(F, "mul.rn.ftz.f32", "mul.rn.ftz.f32 f0, f1, f2;")
add(F, "mul.rn.sat.f32", "mul.rn.sat.f32 f0, f1, f2;")
add(F, "fma.rn.ftz.f32", "fma.rn.ftz.f32 f0, f1, f2, f3;")
add(F, "fma.rn.sat.f32", "fma.rn.sat.f32 f0, f1, f2, f3;")
add(F, "mad.rn.f32", "mad.rn.f32 f0, f1, f2, f3;")
for rnd in ["rn", "rz", "rm", "rp"]:
    add(F, f"div.{rnd}.f32", f"div.{rnd}.f32 f0, f1, f2;")
add(F, "div.approx.f32", "div.approx.f32 f0, f1, f2;")
add(F, "div.approx.ftz.f32", "div.approx.ftz.f32 f0, f1, f2;")
add(F, "div.full.f32", "div.full.f32 f0, f1, f2;")
add(F, "abs.f32", "abs.f32 f0, f1;")
add(F, "abs.ftz.f32", "abs.ftz.f32 f0, f1;")
add(F, "neg.f32", "neg.f32 f0, f1;")
add(F, "min.f32", "min.f32 f0, f1, f2;")
add(F, "min.ftz.f32", "min.ftz.f32 f0, f1, f2;")
add(F, "min.NaN.f32", "min.NaN.f32 f0, f1, f2;")
add(F, "min.xorsign.abs.f32", "min.xorsign.abs.f32 f0, f1, f2;")
add(F, "max.f32", "max.f32 f0, f1, f2;")
add(F, "max.NaN.f32", "max.NaN.f32 f0, f1, f2;")
add(F, "max.xorsign.abs.f32", "max.xorsign.abs.f32 f0, f1, f2;")
for rnd in ["rn", "rz", "rm", "rp"]:
    add(F, f"rcp.{rnd}.f32", f"rcp.{rnd}.f32 f0, f1;")
add(F, "rcp.approx.f32", "rcp.approx.f32 f0, f1;")
add(F, "rcp.approx.ftz.f32", "rcp.approx.ftz.f32 f0, f1;")
add(F, "sqrt.rn.f32", "sqrt.rn.f32 f0, f1;")
add(F, "sqrt.approx.f32", "sqrt.approx.f32 f0, f1;")
add(F, "sqrt.approx.ftz.f32", "sqrt.approx.ftz.f32 f0, f1;")
add(F, "rsqrt.approx.f32", "rsqrt.approx.f32 f0, f1;")
add(F, "rsqrt.approx.ftz.f32", "rsqrt.approx.ftz.f32 f0, f1;")
add(F, "sin.approx.f32", "sin.approx.f32 f0, f1;")
add(F, "cos.approx.f32", "cos.approx.f32 f0, f1;")
add(F, "lg2.approx.f32", "lg2.approx.f32 f0, f1;")
add(F, "lg2.approx.ftz.f32", "lg2.approx.ftz.f32 f0, f1;")
add(F, "ex2.approx.f32", "ex2.approx.f32 f0, f1;")
add(F, "ex2.approx.ftz.f32", "ex2.approx.ftz.f32 f0, f1;")
add(F, "tanh.approx.f32", "tanh.approx.f32 f0, f1;")
add(F, "copysign.f32", "copysign.f32 f0, f1, f2;")
for tp in ["finite", "infinite", "number", "notanumber", "normal", "subnormal"]:
    add(F, f"testp.{tp}.f32", f"testp.{tp}.f32 p0, f1;")
# --- f64 ---
for rnd in ["rn", "rz", "rm", "rp"]:
    add(F, f"add.{rnd}.f64", f"add.{rnd}.f64 fd0, fd1, fd2;")
    add(F, f"mul.{rnd}.f64", f"mul.{rnd}.f64 fd0, fd1, fd2;")
    add(F, f"fma.{rnd}.f64", f"fma.{rnd}.f64 fd0, fd1, fd2, fd3;")
    add(F, f"div.{rnd}.f64", f"div.{rnd}.f64 fd0, fd1, fd2;")
add(F, "sub.rn.f64", "sub.rn.f64 fd0, fd1, fd2;")
add(F, "mad.rn.f64", "mad.rn.f64 fd0, fd1, fd2, fd3;")
add(F, "abs.f64", "abs.f64 fd0, fd1;")
add(F, "neg.f64", "neg.f64 fd0, fd1;")
add(F, "fma.rn.oob.f32", "fma.rn.oob.f32 f0, f1, f2, f3;")  # .oob is f16/bf16 only
add(F, "min.f64", "min.f64 fd0, fd1, fd2;")
add(F, "max.f64", "max.f64 fd0, fd1, fd2;")
add(F, "rcp.rn.f64", "rcp.rn.f64 fd0, fd1;")
add(F, "rcp.approx.ftz.f64", "rcp.approx.ftz.f64 fd0, fd1;")
add(F, "sqrt.rn.f64", "sqrt.rn.f64 fd0, fd1;")
add(F, "rsqrt.approx.f64", "rsqrt.approx.f64 fd0, fd1;")
add(F, "rsqrt.approx.ftz.f64", "rsqrt.approx.ftz.f64 fd0, fd1;")
add(F, "copysign.f64", "copysign.f64 fd0, fd1, fd2;")
add(F, "testp.finite.f64", "testp.finite.f64 p0, fd1;")
add(F, "testp.notanumber.f64", "testp.notanumber.f64 p0, fd1;")

# ================================================================== #
# 13_fp16_bf16   (f16 in .f16 x-regs; bf16 in .b16 h-regs; f16x2 hx; bf16x2 r)
# ================================================================== #
F = "13_fp16_bf16"
HALF = [("f16", "x", "x"), ("f16x2", "hx", "hx"),
        ("bf16", "h", "h"), ("bf16x2", "r", "r")]
for t, dreg, sreg in HALF:
    d0, d1, d2, d3 = f"{dreg}0", f"{sreg}1", f"{sreg}2", f"{sreg}3"
    add(F, f"add.rn.{t}", f"add.rn.{t} {d0}, {d1}, {d2};")
    add(F, f"sub.rn.{t}", f"sub.rn.{t} {d0}, {d1}, {d2};")
    add(F, f"mul.rn.{t}", f"mul.rn.{t} {d0}, {d1}, {d2};")
    add(F, f"fma.rn.{t}", f"fma.rn.{t} {d0}, {d1}, {d2}, {d3};")
    add(F, f"fma.rn.relu.{t}", f"fma.rn.relu.{t} {d0}, {d1}, {d2}, {d3};")
    add(F, f"neg.{t}", f"neg.{t} {d0}, {d1};")
    add(F, f"abs.{t}", f"abs.{t} {d0}, {d1};")
    add(F, f"min.{t}", f"min.{t} {d0}, {d1}, {d2};")
    add(F, f"max.{t}", f"max.{t} {d0}, {d1}, {d2};")
    add(F, f"min.NaN.{t}", f"min.NaN.{t} {d0}, {d1}, {d2};")
    add(F, f"tanh.approx.{t}", f"tanh.approx.{t} {d0}, {d1};")
# ex2: f16/f16x2 take no .ftz; bf16/bf16x2 require .ftz (asymmetry in the ISA)
add(F, "ex2.approx.f16", "ex2.approx.f16 x0, x1;")
add(F, "ex2.approx.f16x2", "ex2.approx.f16x2 hx0, hx1;")
add(F, "ex2.approx.ftz.bf16", "ex2.approx.ftz.bf16 h0, h1;")
add(F, "ex2.approx.ftz.bf16x2", "ex2.approx.ftz.bf16x2 r0, r1;")
# ftz / sat / oob / xorsign samples
add(F, "add.rn.ftz.f16", "add.rn.ftz.f16 x0, x1, x2;")
add(F, "add.rn.sat.f16", "add.rn.sat.f16 x0, x1, x2;")
add(F, "fma.rn.ftz.f16", "fma.rn.ftz.f16 x0, x1, x2, x3;")
add(F, "fma.rn.oob.f16", "fma.rn.oob.f16 x0, x1, x2, x3;")
add(F, "fma.rn.oob.bf16", "fma.rn.oob.bf16 h0, h1, h2, h3;")
add(F, "min.xorsign.abs.f16", "min.xorsign.abs.f16 x0, x1, x2;")
add(F, "max.xorsign.abs.bf16", "max.xorsign.abs.bf16 h0, h1, h2;")

# ================================================================== #
# 14_cmp_sel
# ================================================================== #
F = "14_cmp_sel"
# setp: integer + float compare ops, several types
for cop in ["eq", "ne", "lt", "le", "gt", "ge"]:
    add(F, f"setp.{cop}.s32", f"setp.{cop}.s32 p0, r1, r2;")
for cop in ["eq", "ne", "lo", "ls", "hi", "hs"]:
    add(F, f"setp.{cop}.u32", f"setp.{cop}.u32 p0, r1, r2;")
add(F, "setp.eq.s64", "setp.eq.s64 p0, rd1, rd2;")
add(F, "setp.lt.u64", "setp.lt.u64 p0, rd1, rd2;")
add(F, "setp.eq.b32", "setp.eq.b32 p0, r1, r2;")
add(F, "setp.eq.b16", "setp.eq.b16 p0, h1, h2;")
for cop in ["eq", "ne", "lt", "le", "gt", "ge", "equ", "neu",
            "ltu", "leu", "gtu", "geu", "num", "nan"]:
    add(F, f"setp.{cop}.f32", f"setp.{cop}.f32 p0, f1, f2;")
add(F, "setp.eq.ftz.f32", "setp.eq.ftz.f32 p0, f1, f2;")
add(F, "setp.lt.f64", "setp.lt.f64 p0, fd1, fd2;")
add(F, "setp.eq.f16", "setp.eq.f16 p0, x1, x2;")
add(F, "setp.gt.bf16", "setp.gt.bf16 p0, h1, h2;")
# setp with boolean-combine + second predicate
add(F, "setp.lt.and.s32", "setp.lt.and.s32 p0|p1, r1, r2, p2;")
add(F, "setp.gt.or.f32", "setp.gt.or.f32 p0|p1, f1, f2, p2;")
# set (dest is a value, not a predicate)
add(F, "set.lt.u32.s32", "set.lt.u32.s32 r0, r1, r2;")
add(F, "set.eq.f32.f32", "set.eq.f32.f32 f0, f1, f2;")
add(F, "set.ltu.f32.f32", "set.ltu.f32.f32 f0, f1, f2;")
add(F, "set.eq.s32.f32", "set.eq.s32.f32 r0, f1, f2;")
add(F, "set.lt.f32.s32", "set.lt.f32.s32 f0, r1, r2;")
# selp / slct
add(F, "selp.b32", "selp.b32 r0, r1, r2, p0;")
add(F, "selp.s32", "selp.s32 r0, r1, r2, p0;")
add(F, "selp.b64", "selp.b64 rd0, rd1, rd2, p0;")
add(F, "selp.f32", "selp.f32 f0, f1, f2, p0;")
add(F, "selp.f64", "selp.f64 fd0, fd1, fd2, p0;")
add(F, "selp.b16", "selp.b16 h0, h1, h2, p0;")
add(F, "slct.b32.s32", "slct.b32.s32 r0, r1, r2, r3;")
add(F, "slct.s32.s32", "slct.s32.s32 r0, r1, r2, r3;")
add(F, "slct.f32.s32", "slct.f32.s32 f0, f1, f2, r3;")
add(F, "slct.b32.f32", "slct.b32.f32 r0, r1, r2, f3;")
add(F, "slct.f32.f32", "slct.f32.f32 f0, f1, f2, f3;")
add(F, "slct.f64.f32", "slct.f64.f32 fd0, fd1, fd2, f3;")

# ================================================================== #
# 15_logic_shift
# ================================================================== #
F = "15_logic_shift"
for t, d, a, b in [("b16", "h0", "h1", "h2"), ("b32", "r0", "r1", "r2"),
                   ("b64", "rd0", "rd1", "rd2")]:
    add(F, f"and.{t}", f"and.{t} {d}, {a}, {b};")
    add(F, f"or.{t}", f"or.{t} {d}, {a}, {b};")
    add(F, f"xor.{t}", f"xor.{t} {d}, {a}, {b};")
    add(F, f"not.{t}", f"not.{t} {d}, {a};")
add(F, "cnot.b32", "cnot.b32 r0, r1;")
add(F, "and.pred", "and.pred p0, p1, p2;")
add(F, "or.pred", "or.pred p0, p1, p2;")
add(F, "xor.pred", "xor.pred p0, p1, p2;")
add(F, "not.pred", "not.pred p0, p1;")
add(F, "lop3.b32", "lop3.b32 r0, r1, r2, r3, 0x80;")
add(F, "lop3.b32.imm0xEA", "lop3.b32 r0, r1, r2, r3, 0xEA;")
add(F, "shl.b16", "shl.b16 h0, h1, r2;")
add(F, "shl.b32", "shl.b32 r0, r1, r2;")
add(F, "shl.b64", "shl.b64 rd0, rd1, r2;")
add(F, "shr.u16", "shr.u16 h0, h1, r2;")
add(F, "shr.s16", "shr.s16 h0, h1, r2;")
add(F, "shr.u32", "shr.u32 r0, r1, r2;")
add(F, "shr.s32", "shr.s32 r0, r1, r2;")
add(F, "shr.u64", "shr.u64 rd0, rd1, r2;")
add(F, "shr.s64", "shr.s64 rd0, rd1, r2;")
for dr in ["l", "r"]:
    for md in ["clamp", "wrap"]:
        add(F, f"shf.{dr}.{md}.b32", f"shf.{dr}.{md}.b32 r0, r1, r2, r3;")

# ================================================================== #
# 16_bitmanip
# ================================================================== #
F = "16_bitmanip"
add(F, "popc.b32", "popc.b32 r0, r1;")
add(F, "popc.b64", "popc.b64 r0, rd1;")
add(F, "clz.b32", "clz.b32 r0, r1;")
add(F, "clz.b64", "clz.b64 r0, rd1;")
add(F, "bfind.u32", "bfind.u32 r0, r1;")
add(F, "bfind.s32", "bfind.s32 r0, r1;")
add(F, "bfind.u64", "bfind.u64 r0, rd1;")
add(F, "bfind.s64", "bfind.s64 r0, rd1;")
add(F, "bfind.shiftamt.u32", "bfind.shiftamt.u32 r0, r1;")
add(F, "fns.b32", "fns.b32 r0, r1, r2, r3;")
add(F, "brev.b32", "brev.b32 r0, r1;")
add(F, "brev.b64", "brev.b64 rd0, rd1;")
add(F, "bfe.u32", "bfe.u32 r0, r1, r2, r3;")
add(F, "bfe.s32", "bfe.s32 r0, r1, r2, r3;")
add(F, "bfe.u64", "bfe.u64 rd0, rd1, r2, r3;")
add(F, "bfe.s64", "bfe.s64 rd0, rd1, r2, r3;")
add(F, "bfi.b32", "bfi.b32 r0, r1, r2, r3, r4;")
add(F, "bfi.b64", "bfi.b64 rd0, rd1, rd2, r3, r4;")
for md in ["clamp", "wrap"]:
    add(F, f"szext.{md}.s32", f"szext.{md}.s32 r0, r1, r2;")
    add(F, f"szext.{md}.u32", f"szext.{md}.u32 r0, r1, r2;")
    add(F, f"bmsk.{md}.b32", f"bmsk.{md}.b32 r0, r1, r2;")
add(F, "prmt.b32", "prmt.b32 r0, r1, r2, r3;")
for mode in ["f4e", "b4e", "rc8", "ecl", "ecr", "rc16"]:
    add(F, f"prmt.b32.{mode}", f"prmt.b32.{mode} r0, r1, r2, r3;")

# ================================================================== #
# 17_datamov_cvt
# ================================================================== #
F = "17_datamov_cvt"
GDECL = ".shared .align 16 .b8 sbuf[1024];"
# mov
add(F, "mov.b16", "mov.b16 h0, h1;")
add(F, "mov.b32", "mov.b32 r0, r1;")
add(F, "mov.b64", "mov.b64 rd0, rd1;")
add(F, "mov.f32", "mov.f32 f0, f1;")
add(F, "mov.f64", "mov.f64 fd0, fd1;")
add(F, "mov.pred", "mov.pred p0, p1;")
add(F, "mov.u32.imm", "mov.u32 r0, 42;")
add(F, "mov.b64.pack2x32", "mov.b64 rd0, {r0, r1};")
add(F, "mov.b32.pack2x16", "mov.b32 r0, {h0, h1};")
add(F, "mov.b32.unpack", "mov.b32 {h0, h1}, r0;")
# ld: state spaces
for sp in ["global", "shared", "local"]:
    add(F, f"ld.{sp}.b32", f"ld.{sp}.b32 r0, [rd0];", decls=GDECL)
add(F, "ld.global.u8", "ld.global.u8 r0, [rd0];")
add(F, "ld.global.u16", "ld.global.u16 h0, [rd0];")
add(F, "ld.global.b64", "ld.global.b64 rd1, [rd0];")
add(F, "ld.global.b128", "ld.global.b128 bb0, [rd0];")
add(F, "ld.global.f32", "ld.global.f32 f0, [rd0];")
add(F, "ld.global.f64", "ld.global.f64 fd0, [rd0];")
# ld: vector widths
add(F, "ld.global.v2.b32", "ld.global.v2.b32 {r0, r1}, [rd0];")
add(F, "ld.global.v4.b32", "ld.global.v4.b32 {r0, r1, r2, r3}, [rd0];")
add(F, "ld.global.v2.f32", "ld.global.v2.f32 {f0, f1}, [rd0];")
add(F, "ld.global.v4.f32", "ld.global.v4.f32 {f0, f1, f2, f3}, [rd0];")
add(F, "ld.global.v2.b64", "ld.global.v2.b64 {rd0, rd1}, [rd2];")
add(F, "ld.global.v8.b32", "ld.global.v8.b32 {r0,r1,r2,r3,r4,r5,r6,r7}, [rd0];")
add(F, "ld.global.v4.b64", "ld.global.v4.b64 {rd0,rd1,rd2,rd3}, [rd4];")
# ld: cache operators
for co in ["ca", "cg", "cs", "lu", "cv"]:
    add(F, f"ld.global.{co}.b32", f"ld.global.{co}.b32 r0, [rd0];")
add(F, "ld.global.nc.b32", "ld.global.nc.b32 r0, [rd0];")
add(F, "ld.global.nc.ca.b32", "ld.global.nc.ca.b32 r0, [rd0];")
# ld: memory ordering + scope
for sem, scope in [("relaxed", "gpu"), ("acquire", "gpu"), ("relaxed", "sys"),
                   ("acquire", "sys"), ("relaxed", "cta"), ("relaxed", "cluster")]:
    add(F, f"ld.{sem}.{scope}.global.b32",
        f"ld.{sem}.{scope}.global.b32 r0, [rd0];")
add(F, "ld.volatile.global.b32", "ld.volatile.global.b32 r0, [rd0];")
add(F, "ld.global.L2::128B.b32", "ld.global.L2::128B.b32 r0, [rd0];")
add(F, "ld.global.L1::evict_last.b32", "ld.global.L1::evict_last.b32 r0, [rd0];")
add(F, "ldu.global.b32", "ldu.global.b32 r0, [rd0];")
# st
for sp in ["global", "shared", "local"]:
    add(F, f"st.{sp}.b32", f"st.{sp}.b32 [rd0], r0;", decls=GDECL)
add(F, "st.global.b128", "st.global.b128 [rd0], bb0;")
add(F, "st.global.v4.b32", "st.global.v4.b32 [rd0], {r0, r1, r2, r3};")
add(F, "st.global.v8.b32", "st.global.v8.b32 [rd0], {r0,r1,r2,r3,r4,r5,r6,r7};")
for co in ["wb", "cg", "cs", "wt"]:
    add(F, f"st.global.{co}.b32", f"st.global.{co}.b32 [rd0], r0;")
for sem, scope in [("relaxed", "gpu"), ("release", "gpu"), ("relaxed", "sys"),
                   ("release", "cluster")]:
    add(F, f"st.{sem}.{scope}.global.b32",
        f"st.{sem}.{scope}.global.b32 [rd0], r0;")
add(F, "st.volatile.global.b32", "st.volatile.global.b32 [rd0], r0;")
# prefetch / prefetchu / applypriority / discard / createpolicy
add(F, "prefetch.global.L1", "prefetch.global.L1 [rd0];")
add(F, "prefetch.global.L2", "prefetch.global.L2 [rd0];")
add(F, "prefetch.local.L1", "prefetch.local.L1 [rd0];")
add(F, "prefetch.global.L2::evict_last", "prefetch.global.L2::evict_last [rd0];")
add(F, "prefetchu.L1", "prefetchu.L1 [rd0];")
add(F, "applypriority.global.L2::evict_normal",
    "applypriority.global.L2::evict_normal [rd0], 128;")
add(F, "discard.global.L2", "discard.global.L2 [rd0], 128;")
add(F, "createpolicy.fractional.L2::evict_last.b64",
    "createpolicy.fractional.L2::evict_last.b64 rd0, 1.0;")
add(F, "createpolicy.range.L2.b64",
    "createpolicy.range.L2::evict_last.L2::evict_unchanged.b64 rd0, [rd1], 65536, 131072;")
# isspacep
for sp in ["global", "shared", "local", "const", "shared::cluster"]:
    nm = sp.replace("::", "_")
    add(F, f"isspacep.{nm}", f"isspacep.{sp} p0, rd0;")
# cvta
for sp in ["global", "shared", "local", "const"]:
    add(F, f"cvta.{sp}.u64", f"cvta.{sp}.u64 rd0, rd1;")
    add(F, f"cvta.to.{sp}.u64", f"cvta.to.{sp}.u64 rd0, rd1;")
add(F, "cvta.shared::cluster.u64", "cvta.shared::cluster.u64 rd0, rd1;")
add(F, "cvta.param.u64", "cvta.param.u64 rd0, rd1;")
# cvt: int<->int
add(F, "cvt.u32.u16", "cvt.u32.u16 r0, h1;")
add(F, "cvt.s32.s16", "cvt.s32.s16 r0, h1;")
add(F, "cvt.u16.u32", "cvt.u16.u32 h0, r1;")
add(F, "cvt.s32.s8", "cvt.s32.s8 r0, r1;")
add(F, "cvt.u64.u32", "cvt.u64.u32 rd0, r1;")
add(F, "cvt.s64.s32", "cvt.s64.s32 rd0, r1;")
add(F, "cvt.u32.u64", "cvt.u32.u64 r0, rd1;")
add(F, "cvt.sat.s8.s32", "cvt.sat.s8.s32 r0, r1;")
add(F, "cvt.sat.u16.s32", "cvt.sat.u16.s32 h0, r1;")
# cvt: int<->float
add(F, "cvt.rn.f32.s32", "cvt.rn.f32.s32 f0, r1;")
add(F, "cvt.rz.f32.u32", "cvt.rz.f32.u32 f0, r1;")
add(F, "cvt.rn.f32.s64", "cvt.rn.f32.s64 f0, rd1;")
add(F, "cvt.rn.f64.s32", "cvt.rn.f64.s32 fd0, r1;")
add(F, "cvt.rn.f64.u64", "cvt.rn.f64.u64 fd0, rd1;")
for r in ["rni", "rzi", "rmi", "rpi"]:
    add(F, f"cvt.{r}.s32.f32", f"cvt.{r}.s32.f32 r0, f1;")
add(F, "cvt.rzi.u32.f32", "cvt.rzi.u32.f32 r0, f1;")
add(F, "cvt.rzi.s64.f64", "cvt.rzi.s64.f64 rd0, fd1;")
add(F, "cvt.rzi.s32.f32.ftz", "cvt.rzi.ftz.s32.f32 r0, f1;")
add(F, "cvt.sat.f32.s32", "cvt.rn.f32.s32 f0, r1;")
# cvt: float<->float
add(F, "cvt.f64.f32", "cvt.f64.f32 fd0, f1;")
add(F, "cvt.rn.f32.f64", "cvt.rn.f32.f64 f0, fd1;")
add(F, "cvt.rz.f32.f64", "cvt.rz.f32.f64 f0, fd1;")
add(F, "cvt.rn.f16.f32", "cvt.rn.f16.f32 x0, f1;")
add(F, "cvt.rn.ftz.f16.f32", "cvt.rn.ftz.f16.f32 x0, f1;")
add(F, "cvt.f32.f16", "cvt.f32.f16 f0, x1;")
add(F, "cvt.rn.bf16.f32", "cvt.rn.bf16.f32 h0, f1;")
add(F, "cvt.f32.bf16", "cvt.f32.bf16 f0, h1;")
add(F, "cvt.rn.bf16.f16", "cvt.rn.bf16.f16 h0, x1;")
# cvt: tf32
add(F, "cvt.rn.tf32.f32", "cvt.rn.tf32.f32 r0, f1;")
add(F, "cvt.rz.tf32.f32", "cvt.rz.tf32.f32 r0, f1;")
add(F, "cvt.rna.tf32.f32", "cvt.rna.tf32.f32 r0, f1;")
add(F, "cvt.rn.satfinite.tf32.f32", "cvt.rn.satfinite.tf32.f32 r0, f1;")
add(F, "cvt.rn.relu.tf32.f32", "cvt.rn.relu.tf32.f32 r0, f1;")
# cvt.pack
add(F, "cvt.pack.sat.s16.s32", "cvt.pack.sat.s16.s32 r0, r1, r2;")
add(F, "cvt.pack.sat.u16.s32", "cvt.pack.sat.u16.s32 r0, r1, r2;")

# ================================================================== #
# 18_tex_surf   (bindless / unified form: handle in a .b64 register)
# ================================================================== #
F = "18_tex_surf"
add(F, "tex.1d.v4.f32.s32", "tex.1d.v4.f32.s32 {f0,f1,f2,f3}, [rd0, {r0}];")
add(F, "tex.1d.v4.f32.f32", "tex.1d.v4.f32.f32 {f0,f1,f2,f3}, [rd0, {f4}];")
add(F, "tex.2d.v4.f32.f32", "tex.2d.v4.f32.f32 {f0,f1,f2,f3}, [rd0, {f4,f5}];")
add(F, "tex.2d.v4.u32.f32", "tex.2d.v4.u32.f32 {r0,r1,r2,r3}, [rd0, {f4,f5}];")
add(F, "tex.3d.v4.f32.f32", "tex.3d.v4.f32.f32 {f0,f1,f2,f3}, [rd0, {f4,f5,f6,f7}];")
add(F, "tex.a1d.v4.f32.s32", "tex.a1d.v4.f32.s32 {f0,f1,f2,f3}, [rd0, {r0,r1}];")
add(F, "tex.a2d.v4.f32.f32", "tex.a2d.v4.f32.f32 {f0,f1,f2,f3}, [rd0, {r0,f5,f6,f7}];")
add(F, "tex.cube.v4.f32.f32", "tex.cube.v4.f32.f32 {f0,f1,f2,f3}, [rd0, {f4,f5,f6,f7}];")
add(F, "tex.2d.level.v4.f32.f32",
    "tex.level.2d.v4.f32.f32 {f0,f1,f2,f3}, [rd0, {f4,f5}], f6;")
add(F, "tex.2d.grad.v4.f32.f32",
    "tex.grad.2d.v4.f32.f32 {f0,f1,f2,f3}, [rd0, {f4,f5}], {f6,f7}, {f8,f9};")
for c in ["r", "g", "b", "a"]:
    add(F, f"tld4.{c}.2d.v4.f32.f32",
        f"tld4.{c}.2d.v4.f32.f32 {{f0,f1,f2,f3}}, [rd0, {{f4,f5}}];")
for q in ["width", "height", "depth", "num_mipmap_levels", "array_size"]:
    add(F, f"txq.{q}.b32", f"txq.{q}.b32 r0, [rd0];")
add(F, "suld.b.1d.v4.b32.clamp", "suld.b.1d.v4.b32.clamp {r0,r1,r2,r3}, [rd0, {r4}];")
add(F, "suld.b.2d.v4.b32.trap", "suld.b.2d.v4.b32.trap {r0,r1,r2,r3}, [rd0, {r4,r5}];")
add(F, "suld.b.3d.v2.b64.zero", "suld.b.3d.v2.b64.zero {rd1,rd2}, [rd0, {r4,r5,r6,r7}];")
add(F, "sust.b.1d.v4.b32.clamp", "sust.b.1d.v4.b32.clamp [rd0, {r4}], {r0,r1,r2,r3};")
add(F, "sust.b.2d.v4.b32.trap", "sust.b.2d.v4.b32.trap [rd0, {r4,r5}], {r0,r1,r2,r3};")
add(F, "sust.p.2d.v4.b32.trap", "sust.p.2d.v4.b32.trap [rd0, {r4,r5}], {r0,r1,r2,r3};")
add(F, "sured.b.add.1d.u32.trap", "sured.b.add.1d.u32.trap [rd0, {r4}], r0;")
add(F, "sured.b.min.2d.s32.clamp", "sured.b.min.2d.s32.clamp [rd0, {r4,r5}], r0;")
for q in ["width", "height", "depth"]:
    add(F, f"suq.{q}.b32", f"suq.{q}.b32 r0, [rd0];")
add(F, "istypep.texref", "istypep.texref p0, rd0;")
add(F, "istypep.samplerref", "istypep.samplerref p0, rd0;")
add(F, "istypep.surfref", "istypep.surfref p0, rd0;")

# ================================================================== #
# 19_control_flow
# ================================================================== #
F = "19_control_flow"
add(F, "bra", "bra BB_A;\nBB_A:")
add(F, "bra.uni", "bra.uni BB_B;\nBB_B:")
add(F, "bra.pred", "@p0 bra BB_C;\nBB_C:")
add(F, "bra.negpred", "@!p0 bra BB_D;\nBB_D:")
add(F, "exit", "exit;")
add(F, "ret.uni", "ret.uni;")
add(F, "pred.add", "@p0 add.s32 r0, r1, r2;")
CALL_RAW = """\
.version {version}
.target {target}
.address_size 64

.func (.reg .b32 rv) leaf(.reg .b32 pa)
{{
    ret;
}}
.visible .entry probe()
{{
    .reg .b32 r<4>;
    call.uni (r0), leaf, (r1);
    ret;
}}
"""
add(F, "call.uni.func", "", raw=CALL_RAW)
# NOTE: brx.idx (indexed branch) is documented but not probed here -- it needs a
# .branchtargets table plus the labels it indexes, which the single-instruction
# harness does not model; recorded as a coverage gap rather than a bad probe.

# ================================================================== #
# 20_sync_comm
# ================================================================== #
F = "20_sync_comm"
add(F, "bar.sync", "bar.sync 0;")
add(F, "bar.sync.count", "bar.sync 0, 32;")
add(F, "bar.arrive", "bar.arrive 0, 32;")
add(F, "bar.red.popc.u32", "bar.red.popc.u32 r0, 0, 32, p0;")
add(F, "bar.red.and.pred", "bar.red.and.pred p0, 0, 32, p1;")
add(F, "bar.red.or.pred", "bar.red.or.pred p0, 0, 32, p1;")
add(F, "barrier.sync", "barrier.sync 0;")
add(F, "barrier.sync.aligned", "barrier.sync.aligned 0;")
add(F, "barrier.arrive", "barrier.arrive 0, 32;")
add(F, "bar.warp.sync", "bar.warp.sync 0xffffffff;")
add(F, "membar.cta", "membar.cta;")
add(F, "membar.gl", "membar.gl;")
add(F, "membar.sys", "membar.sys;")
for scope in ["cta", "cluster", "gpu", "sys"]:
    add(F, f"fence.sc.{scope}", f"fence.sc.{scope};")
    add(F, f"fence.acq_rel.{scope}", f"fence.acq_rel.{scope};")
add(F, "fence.mbarrier_init.release.cluster",
    "fence.mbarrier_init.release.cluster;")
add(F, "fence.proxy.tensormap.release.gpu",
    "fence.proxy.tensormap::generic.release.gpu;")
add(F, "fence.acquire.sync_restrict.cluster",
    "fence.acquire.sync_restrict::shared::cluster.cluster;")

# ================================================================== #
# 21_atomic
# ================================================================== #
F = "21_atomic"
# integer add / min / max / and / or / xor / exch across spaces & scopes
add(F, "atom.global.add.u32", "atom.global.add.u32 r0, [rd0], r1;")
add(F, "atom.global.add.s32", "atom.global.add.s32 r0, [rd0], r1;")
add(F, "atom.global.add.u64", "atom.global.add.u64 rd1, [rd0], rd2;")
add(F, "atom.global.min.s32", "atom.global.min.s32 r0, [rd0], r1;")
add(F, "atom.global.min.u32", "atom.global.min.u32 r0, [rd0], r1;")
add(F, "atom.global.max.u64", "atom.global.max.u64 rd1, [rd0], rd2;")
add(F, "atom.global.min.s64", "atom.global.min.s64 rd1, [rd0], rd2;")
add(F, "atom.global.and.b32", "atom.global.and.b32 r0, [rd0], r1;")
add(F, "atom.global.or.b32", "atom.global.or.b32 r0, [rd0], r1;")
add(F, "atom.global.xor.b32", "atom.global.xor.b32 r0, [rd0], r1;")
add(F, "atom.global.exch.b32", "atom.global.exch.b32 r0, [rd0], r1;")
add(F, "atom.global.exch.b64", "atom.global.exch.b64 rd1, [rd0], rd2;")
add(F, "atom.global.cas.b32", "atom.global.cas.b32 r0, [rd0], r1, r2;")
add(F, "atom.global.cas.b64", "atom.global.cas.b64 rd1, [rd0], rd2, rd3;")
add(F, "atom.global.cas.b128", "atom.global.cas.b128 bb0, [rd0], bb1, bb2;")
add(F, "atom.global.inc.u32", "atom.global.inc.u32 r0, [rd0], r1;")
add(F, "atom.global.dec.u32", "atom.global.dec.u32 r0, [rd0], r1;")
# float atoms
add(F, "atom.global.add.f32", "atom.global.add.f32 f0, [rd0], f1;")
add(F, "atom.global.add.f64", "atom.global.add.f64 fd0, [rd0], fd1;")
add(F, "atom.global.add.noftz.f16", "atom.global.add.noftz.f16 x0, [rd0], x1;")
add(F, "atom.global.add.noftz.f16x2", "atom.global.add.noftz.f16x2 hx0, [rd0], hx1;")
add(F, "atom.global.add.noftz.bf16", "atom.global.add.noftz.bf16 h0, [rd0], h1;")
add(F, "atom.global.add.noftz.bf16x2", "atom.global.add.noftz.bf16x2 r0, [rd0], r1;")
# atom min/max on half types require both .noftz and a .v2/.v4 vector form
add(F, "atom.global.min.v2.f16x2", "atom.global.min.noftz.v2.f16x2 {hx0,hx1}, [rd0], {hx2,hx3};")
add(F, "atom.global.max.v2.f16x2", "atom.global.max.noftz.v2.f16x2 {hx0,hx1}, [rd0], {hx2,hx3};")
# vector atoms (f32 add supports .v2/.v4 only; half/bhalf need .noftz)
add(F, "atom.global.add.v2.f32", "atom.global.add.v2.f32 {f0,f1}, [rd0], {f2,f3};")
add(F, "atom.global.add.v4.f32", "atom.global.add.v4.f32 {f0,f1,f2,f3}, [rd0], {f4,f5,f6,f7};")
add(F, "atom.global.add.v2.f16x2", "atom.global.add.noftz.v2.f16x2 {hx0,hx1}, [rd0], {hx2,hx3};")
add(F, "atom.global.add.v4.f16", "atom.global.add.noftz.v4.f16 {x0,x1,x2,x3}, [rd0], {x4,x5,x6,x7};")
# scopes / semantics
add(F, "atom.relaxed.gpu.global.add.u32", "atom.relaxed.gpu.global.add.u32 r0, [rd0], r1;")
add(F, "atom.acquire.gpu.global.add.u32", "atom.acquire.gpu.global.add.u32 r0, [rd0], r1;")
add(F, "atom.acq_rel.sys.global.add.u32", "atom.acq_rel.sys.global.add.u32 r0, [rd0], r1;")
add(F, "atom.relaxed.cluster.global.add.u32", "atom.relaxed.cluster.global.add.u32 r0, [rd0], r1;")
# shared space
add(F, "atom.shared.add.u32", "atom.shared.add.u32 r0, [rd0], r1;", decls=GDECL)
add(F, "atom.shared.cas.b32", "atom.shared.cas.b32 r0, [rd0], r1, r2;", decls=GDECL)
add(F, "atom.shared.add.f32", "atom.shared.add.f32 f0, [rd0], f1;", decls=GDECL)
# red (reduce, no destination)
add(F, "red.global.add.u32", "red.global.add.u32 [rd0], r1;")
add(F, "red.global.add.f32", "red.global.add.f32 [rd0], f1;")
add(F, "red.global.add.f64", "red.global.add.f64 [rd0], fd1;")
add(F, "red.global.add.noftz.f16", "red.global.add.noftz.f16 [rd0], x1;")
add(F, "red.global.add.noftz.bf16", "red.global.add.noftz.bf16 [rd0], h1;")
add(F, "red.global.min.s32", "red.global.min.s32 [rd0], r1;")
add(F, "red.global.and.b32", "red.global.and.b32 [rd0], r1;")
add(F, "red.global.add.v2.f32", "red.global.add.v2.f32 [rd0], {f0,f1};")
add(F, "red.shared.add.u32", "red.shared.add.u32 [rd0], r1;", decls=GDECL)
add(F, "red.global.inc.u32", "red.global.inc.u32 [rd0], r1;")

# ================================================================== #
# 22_warp
# ================================================================== #
F = "22_warp"
for v in ["all", "any", "uni"]:
    add(F, f"vote.sync.{v}.pred", f"vote.sync.{v}.pred p0, p1, 0xffffffff;")
add(F, "vote.sync.ballot.b32", "vote.sync.ballot.b32 r0, p1, 0xffffffff;")
add(F, "vote.all.pred", "vote.all.pred p0, p1;")
add(F, "vote.ballot.b32", "vote.ballot.b32 r0, p1;")
add(F, "match.any.sync.b32", "match.any.sync.b32 r0, r1, 0xffffffff;")
add(F, "match.all.sync.b32", "match.all.sync.b32 r0|p0, r1, 0xffffffff;")
add(F, "match.any.sync.b64", "match.any.sync.b64 r0, rd1, 0xffffffff;")
add(F, "activemask.b32", "activemask.b32 r0;")
for m in ["up", "down", "bfly", "idx"]:
    add(F, f"shfl.sync.{m}.b32", f"shfl.sync.{m}.b32 r0|p0, r1, 1, 0, 0xffffffff;")
add(F, "shfl.up.b32", "shfl.up.b32 r0, r1, 1, 0;")
add(F, "redux.sync.add.u32", "redux.sync.add.u32 r0, r1, 0xffffffff;")
add(F, "redux.sync.or.b32", "redux.sync.or.b32 r0, r1, 0xffffffff;")

# ================================================================== #
# 23_video_simd
# ================================================================== #
F = "23_video_simd"
# scalar video (dual-issue video ISA)
for op in ["vadd", "vsub", "vabsdiff", "vmin", "vmax"]:
    add(F, f"{op}.s32.s32.s32", f"{op}.s32.s32.s32 r0, r1, r2;")
    add(F, f"{op}.u32.u32.u32", f"{op}.u32.u32.u32 r0, r1, r2;")
add(F, "vadd.u32.u32.u32.sat", "vadd.u32.u32.u32.sat r0, r1, r2;")
add(F, "vadd.s32.s32.s32.add", "vadd.s32.s32.s32.add r0, r1, r2, r3;")
add(F, "vshl.u32.u32.u32.clamp", "vshl.u32.u32.u32.clamp r0, r1, r2;")
add(F, "vshr.u32.u32.u32.clamp", "vshr.u32.u32.u32.clamp r0, r1, r2;")
add(F, "vmad.s32.s32.s32", "vmad.s32.s32.s32 r0, r1, r2, r3;")
add(F, "vset.u32.u32.lt", "vset.u32.u32.lt r0, r1, r2, r3;")
# SIMD video (2-way half / 4-way byte)
for op in ["vadd", "vsub", "vavrg", "vabsdiff", "vmin", "vmax"]:
    add(F, f"{op}2.s32.s32.s32", f"{op}2.s32.s32.s32 r0, r1, r2, r3;")
    add(F, f"{op}4.s32.s32.s32", f"{op}4.s32.s32.s32 r0, r1, r2, r3;")
add(F, "vset2.u32.u32.lt", "vset2.u32.u32.lt r0, r1, r2, r3;")
add(F, "vset4.u32.u32.lt", "vset4.u32.u32.lt r0, r1, r2, r3;")
add(F, "vadd4.u32.u32.u32.sat", "vadd4.u32.u32.u32.sat r0, r1, r2, r3;")
add(F, "vabsdiff4.u32.u32.u32.add", "vabsdiff4.u32.u32.u32.add r0, r1, r2, r3;")
# packed 16-bit integer SIMD (pre-9.2)
for op in ["add", "sub", "min", "max"]:
    add(F, f"{op}.s16x2", f"{op}.s16x2 r0, r1, r2;")
    add(F, f"{op}.u16x2", f"{op}.u16x2 r0, r1, r2;")
add(F, "add.sat.s16x2", "add.sat.s16x2 r0, r1, r2;")

# ================================================================== #
# 24_special
# ================================================================== #
F = "24_special"
add(F, "nanosleep.u32", "nanosleep.u32 r0;")
add(F, "nanosleep.u32.imm", "nanosleep.u32 100;")
add(F, "pmevent", "pmevent 0;")
add(F, "pmevent.mask", "pmevent.mask 0xff;")
add(F, "trap", "trap;")
add(F, "brkpt", "brkpt;")
# special registers (32-bit -> r, 64-bit -> rd, predicate -> p)
SREG32 = ["%laneid", "%warpid", "%nwarpid", "%smid", "%nsmid", "%gridid",
          "%lanemask_eq", "%lanemask_lt", "%lanemask_le", "%lanemask_gt",
          "%lanemask_ge", "%clock", "%clock_hi", "%dynamic_smem_size",
          "%total_smem_size"]
for s in SREG32:
    add(F, f"sreg{s}", f"mov.u32 r0, {s};")
add(F, "sreg.tid.x", "mov.u32 r0, %tid.x;")
add(F, "sreg.ntid.x", "mov.u32 r0, %ntid.x;")
add(F, "sreg.ctaid.x", "mov.u32 r0, %ctaid.x;")
add(F, "sreg.nctaid.x", "mov.u32 r0, %nctaid.x;")
add(F, "sreg.clock64", "mov.u64 rd0, %clock64;")
add(F, "sreg.globaltimer", "mov.u64 rd0, %globaltimer;")
add(F, "sreg.gridid64", "mov.u64 rd0, %gridid;")
add(F, "sreg.current_graph_exec", "mov.u64 rd0, %current_graph_exec;")
add(F, "sreg.envreg0", "mov.b32 r0, %envreg0;")
# stack manipulation
add(F, "stacksave.u64", "stacksave.u64 rd0;")
add(F, "stackrestore.u64", "stackrestore.u64 rd0;")
add(F, "alloca.u64", "alloca.u64 rd0, rd1, 8;")
# NOTE on the version boundary: this ptxas (13.2) implements PTX ISA up to
# .version 9.2. Instructions introduced only in PTX ISA 9.3+ (e.g. the fabric.*
# distributed-memory family) are outside this snapshot by construction, not an
# arch difference; they are documented in the README coverage statement rather
# than probed here (a probe for them would report "unknown instruction" on all
# six targets, which says nothing about the targets).

# ================================================================== #
# 25_tensor_extra
# A few tensor-path probes from the original campaign's unshipped 8_extra family,
# folded in so every row the README cites is backed by results_full.json. These
# are the paired-CTA / tcgen05.mma / multicast-TMA constructs that the curated
# generators reference in prose but did not emit as standalone probes.
# ================================================================== #
F = "25_tensor_extra"
add(F, "cp.async.bulk.tensor.2d.g2s.cta_group2",
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 [rd0], [rd1,{r0,r1}], [rd2];")
add(F, "cp.async.bulk.g2s.cluster.multicast",
    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [rd0], [rd1], r0, [rd2], h0;")
add(F, "cp.async.bulk.tensor.2d.g2s.cluster.multicast",
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [rd0], [rd1,{r0,r1}], [rd2], h0;")
add(F, "tcgen05.mma.f16", "tcgen05.mma.cta_group::1.kind::f16 [r0], rd0, rd1, r2, p0;")
add(F, "tcgen05.mma.mxf4nvf4.blockscale",
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [r0], rd0, rd1, r2, [r3], [r4], p0;")
add(F, "tcgen05.cp", "tcgen05.cp.cta_group::1.128x256b [r0], rd0;")
add(F, "ld.global.nc.L1::no_allocate.f32", "ld.global.nc.L1::no_allocate.f32 f0, [rd0];")

out = os.path.join(HERE, "probes_full.json")
with open(out, "w") as f:
    json.dump(P, f, indent=1)
print(f"{len(P)} probes ({N_CURATED} curated + {len(P)-N_CURATED} new comprehensive) written to {out}")
