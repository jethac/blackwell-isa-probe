#!/usr/bin/env python3
"""SASS-level camp analysis: compile every probe for sm_100a (datacenter),
sm_120a and sm_121a (consumer), disassemble all three, and diff the SASS.

The compile matrix answers "does it assemble"; this answers the deeper question
"does it lower to the SAME machine code across the datacenter vs consumer camps."
Splits that are invisible at compile-accept (a BPT.TRAP stub, an error-barrier
synchronous fallback, a different opcode sequence) show up here.

Usage:
    PTXAS=<ptxas13.3> CUOBJ=<cuobjdump> python harness/sass_camp_analysis.py \
        generators/probes_full.json generators/probes_ptx93.json

Writes results/sass_camp_report.json and prints a summary. Classification per
probe (over the SASS opcode sequence, operands stripped):
  identical            sm100 == sm120 == sm121
  camp-split:trap      consumer differs AND consumer emits BPT (non-functional)
  camp-split:errbar    consumer differs AND consumer emits ERRBAR/CGAERRBAR
  camp-split:opcode    consumer differs from datacenter, other opcodes
  intra-consumer       sm120 != sm121 (rare; notable)
  no-sass              all three emit only boilerplate / DCE to nothing
  compile-fail         ptxas rejected or crashed on >=1 target (ICEs land here)
"""
import sys, os, json, subprocess, re, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import runner

PTXAS = os.environ.get("PTXAS", "ptxas")
CUOBJ = os.environ.get("CUOBJ", "cuobjdump")
ARCHES = ["sm_100a", "sm_120a", "sm_121a"]
# pure setup / address-arithmetic / control boilerplate: ignored when deciding
# whether the *meaningful* lowering differs (kept out of the compared sequence).
BOILER = {"LDC","LDCU","ULDC","MOV","UMOV","EXIT","BRA","NOP","RET","S2R","CS2R",
          "IMAD","IADD3","UIADD3","LOP3","PLOP3","SHF","LEA","R2UR","R2P","P2R"}

def opseq(cubin):
    out = subprocess.run([CUOBJ,"-sass",cubin], capture_output=True, text=True).stdout
    seq = []
    for l in out.splitlines():
        m = re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P?\w+\s+)?([A-Z][A-Z0-9_]*)', l)
        if m: seq.append(m.group(1))
    return seq

def compile_seq(pr, arch):
    ptx = runner.build_ptx(pr.get("code",""), pr.get("decls",""), target=arch,
                           version=pr.get("version","9.3"), raw=pr.get("raw"),
                           no_std_decls=pr.get("no_std_decls", False))
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(ptx); p = f.name
    c = p[:-4] + ".cubin"
    try:
        r = subprocess.run([PTXAS,"-arch="+arch,p,"-o",c], capture_output=True,
                           text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        err = (r.stderr or "")
        tag = "ICE" if "Internal compiler error" in err or r.returncode < 0 else "reject"
        return None, tag
    seq = opseq(c)
    try: os.remove(c)
    except OSError: pass
    try: os.remove(p)
    except OSError: pass
    return seq, "ok"

def classify(seqs, status):
    if any(s != "ok" for s in status.values()):
        fails = {a:t for a,t in status.items() if t != "ok"}
        return ("compile-fail", fails)
    meaningful = {a:[o for o in seqs[a] if o not in BOILER] for a in ARCHES}
    dc, c120, c121 = meaningful["sm_100a"], meaningful["sm_120a"], meaningful["sm_121a"]
    if not any(meaningful.values()):
        return ("no-sass", None)
    if dc == c120 == c121:
        return ("identical", None)
    if c120 != c121:
        return ("intra-consumer", {"sm120":c120, "sm121":c121})
    # consumer camp agrees, differs from datacenter
    cons = c120
    if "BPT" in cons:
        return ("camp-split:trap", {"dc":dc, "consumer":cons})
    if "ERRBAR" in cons or "CGAERRBAR" in cons:
        return ("camp-split:errbar", {"dc":dc, "consumer":cons})
    return ("camp-split:opcode", {"dc":dc, "consumer":cons})

def main():
    files = sys.argv[1:] or ["generators/probes_full.json"]
    probes = []
    seen = set()
    for fn in files:
        for pr in json.load(open(fn)):
            k = pr["family"]+"|"+pr["name"]
            if k in seen: continue
            seen.add(k); probes.append(pr)
    report = {}
    from collections import Counter
    cats = Counter()
    for i, pr in enumerate(probes):
        key = pr["family"]+"|"+pr["name"]
        seqs, status = {}, {}
        for a in ARCHES:
            s, st = compile_seq(pr, a); seqs[a] = s or []; status[a] = st
        cat, detail = classify(seqs, status)
        cats[cat] += 1
        if cat not in ("identical","no-sass"):
            report[key] = {"category": cat, "detail": detail}
        if (i+1) % 100 == 0:
            print(f"...{i+1}/{len(probes)}", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump({"summary": dict(cats), "splits": report},
              open("results/sass_camp_report.json","w"), indent=1)
    print("\n=== SASS camp analysis ===")
    for c,n in cats.most_common(): print(f"  {c:22} {n}")
    print(f"\nnon-identical (excl no-sass): {sum(n for c,n in cats.items() if c not in ('identical','no-sass'))}")
    for k,v in report.items():
        if v["category"].startswith("camp-split") or v["category"] in ("intra-consumer","compile-fail"):
            print(f"  [{v['category']:20}] {k}")

if __name__ == "__main__":
    main()
