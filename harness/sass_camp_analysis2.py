#!/usr/bin/env python3
"""Rigorous SASS camp analysis (fixes the three failure modes of the naive diff).

The v1 sweep (sass_camp_analysis.py) diffed raw opcode SETS over *bare* compile
probes. That over-/under-reports three ways (see results/SASS_CAMP_ANALYSIS.md C):
  1. DCE blind spot   - bare ops with unused outputs vanish (no-sass != identical)
  2. undefined inputs - divergent dead-code scaffolding (rcp VIADD, atom CREDUX)
  3. selection noise  - IADD3 vs IADD is the same op, different encoding

This version fixes all three:
  * SINK-WRAP every probe: every register the op reads is initialized from a
    volatile global load (defined, non-constant), and the destination is stored
    to global. Nothing DCEs; no undefined inputs. (Side-effecting ops - stores,
    atomics, async/fabric/cluster - need no dest store; they can't DCE.)
  * Classify by SEMANTIC MARKERS, not raw opcode sets. A probe is a real camp
    split iff the set of capability-marker opcodes differs across camps:
    traps (BPT), software trampolines (CALL in a leaf kernel), error-barrier
    fallbacks (ERRBAR/CGAERRBAR), or a different tensor/TMA/reduce instruction
    CLASS. Benign selection differences (IADD3/IADD, FADD forms) carry no marker
    and never register.

Usage:
    PTXAS=<ptxas13.3> CUOBJ=<cuobjdump> python harness/sass_camp_analysis2.py \
        generators/probes_full.json generators/probes_ptx93.json
Writes results/sass_camp_report2.json + prints the verified split list.
"""
import sys, os, json, subprocess, re, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import runner

PTXAS = os.environ.get("PTXAS", "ptxas")
CUOBJ = os.environ.get("CUOBJ", "cuobjdump")
ARCHES = ["sm_100a", "sm_120a", "sm_121a"]

# capability markers: a difference in ANY of these across camps = a real split.
# (Deliberately excludes plain arithmetic/mov/control, which carry selection
# noise but no capability meaning.)
MARKERS = {
    "BPT",                                             # trap stub
    "CALL", "CALL.ABS", "CALL.REL",                    # software trampoline
    "ERRBAR", "CGAERRBAR",                             # error-barrier fallback
    # tensor MMA classes (native vs emulated vs datacenter)
    "HMMA", "QMMA", "IMMA", "BMMA", "OMMA", "UTCMMA", "UTCQMMA", "UTCHMMA",
    # TMA / bulk async copy
    "UBLKCP", "UTMALDG", "UTMASTG", "UTMACMDFLUSH", "UBLKPF", "LDGSTS",
    # distributed reduce / multicast mechanisms (async UREDGR vs sync REDG, etc.)
    "UREDGR", "REDG", "REDGMC", "LDGMC", "CREDUX",
    # cluster machinery
    "CGABAR", "UCGABAR",
}

REG_PREFIXES = ["fd","rd","hx","bb","sfa","sfb","f","r","a","b","h","x","q","e","p"]
REG_RE = re.compile(r"\b(fd|rd|hx|bb|sfa|sfb|f|r|a|b|h|x|q|e|p)(\d*)\b")
# how to give each register class a DEFINED value, and how to store it
LD = {"fd":("f64","fds"),"rd":("u64","rds"),"hx":("b32","rs"),"bb":("b128",None),
      "sfa":("b32","rs"),"sfb":("b32","rs"),"f":("f32","fs"),"r":("u32","rs"),
      "a":("b32","rs"),"b":("b32","rs"),"h":("u16","hs"),"x":("b16","hs"),
      "q":("u8","qs"),"e":("b32","rs")}
ST = {"fd":"f64","rd":"u64","hx":"b32","bb":"b128","sfa":"b32","sfb":"b32",
      "f":"f32","r":"u32","a":"b32","b":"b32","h":"u16","x":"b16","q":"u8","e":"b32"}

PROLOG = """\
.version {version}
.target {target}
.address_size 64
.visible .entry sink(.param .u64 po, .param .u64 pin)
{{
{decls}
    .reg .b64 %base, %out;
    .reg .b32 rs; .reg .b64 rds; .reg .b16 hs; .reg .f32 fs; .reg .f64 fds; .reg .b8 qs;
    .reg .pred %pt;
    ld.param.u64 %out, [po];  cvta.to.global.u64 %out, %out;
    ld.param.u64 %base, [pin]; cvta.to.global.u64 %base, %base;
    ld.volatile.global.u32 rs, [%base];
    ld.volatile.global.u64 rds, [%base+8];
    ld.volatile.global.u16 hs, [%base+16];
    ld.volatile.global.f32 fs, [%base+24];
    ld.volatile.global.f64 fds, [%base+32];
    ld.volatile.global.u8  qs, [%base+40];
    setp.ne.u32 %pt, rs, 0;
"""

def reg_class(tok):
    for p in REG_PREFIXES:
        if tok.startswith(p) and (tok[len(p):].isdigit() or tok in ("sfa","sfb") or (p=="e" and tok=="e0")):
            return p
    return None

def sink_ptx(pr, target):
    code = pr.get("code","").strip()
    if not code or pr.get("raw"):
        return None  # raw/no-code probes: not sinkable generically
    body = code.rstrip(";").split(";")[0] + ";"   # first statement only
    # collect register tokens; skip anything inside the instruction mnemonic
    ops = body.split(None, 1)
    operand_str = ops[1] if len(ops) > 1 else ""
    toks = {}
    for m in REG_RE.finditer(operand_str):
        t = m.group(0)
        c = reg_class(t)
        if c: toks[t] = c
    # destination = first top-level operand's register(s)
    first = re.split(r",(?![^{]*})", operand_str.strip())[0].strip()
    side_effecting = first.startswith("[")
    dests = []
    if not side_effecting:
        for m in REG_RE.finditer(first):
            if reg_class(m.group(0)): dests.append(m.group(0))
    # build defines for every read register
    defines = []
    for t, c in toks.items():
        if c == "p":
            defines.append(f"    setp.ne.u32 {t}, rs, 0;")
        elif c == "bb":
            defines.append(f"    ld.volatile.global.b128 {t}, [%base+48];")
        else:
            _, seed = LD[c]
            defines.append(f"    mov.{ 'b16' if c in ('h','x') else ('b32' if ST[c] in ('b32','u32') else ST[c].replace('u','b')) } {t}, {seed};"
                           if seed else f"    mov.b32 {t}, rs;")
    # store the destination(s)
    stores = []
    for i, d in enumerate(dests):
        c = reg_class(d)
        if c == "p":
            stores.append(f"    selp.b32 rs, 1, 0, {d};\n    st.global.u32 [%out+{i*16}], rs;")
        elif c == "bb":
            stores.append(f"    st.global.b128 [%out+{i*16}], {d};")
        else:
            stores.append(f"    st.global.{ST[c]} [%out+{i*16}], {d};")
    decls = "    " + "\n    ".join(l.strip() for l in runner.STD_DECLS.strip().splitlines())
    extra = "\n".join("    " + l for l in pr.get("decls","").splitlines()) if pr.get("decls") else ""
    ptx = PROLOG.format(version=pr.get("version","9.3"), target=target, decls=decls + ("\n"+extra if extra else ""))
    ptx += "\n".join(defines) + "\n    " + body + "\n" + "\n".join(stores) + "\n    ret;\n}\n"
    return ptx

def markers_of(pr, target):
    ptx = sink_ptx(pr, target)
    if ptx is None: return None, "unsinkable"
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(ptx); p = f.name
    c = p[:-4] + ".cubin"
    try:
        r = subprocess.run([PTXAS,"-arch="+target,"-O3",p,"-o",c], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        return None, ("ICE" if ("Internal compiler error" in (r.stderr or "") or r.returncode<0) else "reject")
    out = subprocess.run([CUOBJ,"-sass",c], capture_output=True, text=True).stdout
    found = set()
    for l in out.splitlines():
        m = re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P?\w+\s+)?([A-Z][A-Z0-9_]*)', l)
        if m and m.group(1) in MARKERS: found.add(m.group(1))
    try: os.remove(c); os.remove(p)
    except OSError: pass
    return found, "ok"

def main():
    files = sys.argv[1:] or ["generators/probes_full.json"]
    probes, seen = [], set()
    for fn in files:
        for pr in json.load(open(fn)):
            k = pr["family"]+"|"+pr["name"]
            if k not in seen: seen.add(k); probes.append(pr)
    from collections import Counter
    cats = Counter(); splits = {}
    for i, pr in enumerate(probes):
        mk, st = {}, {}
        for a in ARCHES:
            m, s = markers_of(pr, a); mk[a] = m; st[a] = s
        if any(s != "ok" for s in st.values()):
            cats["unsinkable/compile-fail"] += 1; continue
        dc, c120, c121 = mk["sm_100a"], mk["sm_120a"], mk["sm_121a"]
        if c120 != c121:
            cats["intra-consumer"] += 1
            splits[pr["family"]+"|"+pr["name"]] = {"cat":"intra-consumer","sm120":sorted(c120),"sm121":sorted(c121)}
        elif dc != c120:
            cats["CAMP-SPLIT"] += 1
            splits[pr["family"]+"|"+pr["name"]] = {"cat":"camp-split",
                "datacenter":sorted(dc),"consumer":sorted(c120),
                "dc_only":sorted(dc-c120),"consumer_only":sorted(c120-dc)}
        else:
            cats["no-marker-diff"] += 1
        if (i+1) % 100 == 0: print(f"...{i+1}/{len(probes)}", flush=True)
    json.dump({"summary":dict(cats),"splits":splits}, open("results/sass_camp_report2.json","w"), indent=1)
    print("\n=== rigorous SASS camp analysis (sink-wrapped, marker-classified) ===")
    for c,n in cats.most_common(): print(f"  {c:26} {n}")
    print("\n=== verified camp-splits (marker difference) ===")
    for k,v in splits.items():
        if v["cat"]=="camp-split":
            print(f"  {k}\n      dc_only={v['dc_only']} consumer_only={v['consumer_only']}")

if __name__ == "__main__":
    main()
