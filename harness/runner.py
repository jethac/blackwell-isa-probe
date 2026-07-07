#!/usr/bin/env python3
"""ISA compile-probe harness for consumer Blackwell audit.

Wraps instruction snippets in a minimal .ptx file and runs ptxas for each
target arch. Records ACCEPT / REJECT(class) per arch. ptxas is the oracle.
"""
import json, os, re, subprocess, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
PROBEDIR = os.path.join(HERE, "probes")
os.makedirs(PROBEDIR, exist_ok=True)

# All six Blackwell targets, in the README's column order.
ARCHES = ["sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_120f", "sm_121a"]

# Per-target minimum PTX ISA .version. sm_110 (Thor) does not exist before PTX
# ISA 9.0, and ptxas reports that coupling with the wrapper-level message
# "PTX .version X does not support .target sm_110a" rather than the
# "requires PTX ISA .version" form the auto-upgrade below keys on -- so we floor
# it here to avoid recording a dialect artifact as an arch answer. (See the
# note in extend_archs.py; that was a real harness bug in the first Thor pass.)
VERSION_FLOOR = {"sm_110a": "9.0"}
MAX_PTX_VERSION = "9.3"   # highest .version ptxas 13.3 accepts (was 9.2 on 13.2)

STD_DECLS = """\
    .reg .f32 f<32>;
    .reg .f64 fd<16>;
    .reg .b32 r<32>;
    .reg .b32 a<8>;
    .reg .b32 b<8>;
    .reg .b32 sfa;
    .reg .b32 sfb;
    .reg .b32 e0;
    .reg .b64 rd<16>;
    .reg .b128 bb<4>;
    .reg .pred p<8>;
    .reg .b16 h<32>;
    .reg .f16 x<8>;
    .reg .f16x2 hx<8>;
    .reg .b8 q<8>;
"""

TEMPLATE = """\
.version {version}
.target {target}
.address_size 64

.visible .entry probe()
{{
{decls}{extra}
    {code}
    ret;
}}
"""

def build_ptx(code, extra_decls="", target="sm_120a", version="8.8", raw=None,
              no_std_decls=False):
    if raw is not None:
        return raw.format(version=version, target=target)
    decls = "" if no_std_decls else STD_DECLS
    if extra_decls:
        extra = "".join("    " + l + "\n" for l in extra_decls.splitlines())
    else:
        extra = ""
    return TEMPLATE.format(version=version, target=target, decls=decls,
                           extra=extra, code=code)

def classify(err):
    """Triage a ptxas reject by its error text. The distinction that matters:
    an ARCH reject ("this target does not have the instruction") is a real ISA
    answer; a SYNTAX/OPERAND/PTXVER reject is a harness/dialect artifact and must
    NOT be recorded as an arch answer. ptxas 13.2's exact phrasings are pinned
    below (verified empirically against V13.2.78)."""
    e = err.lower()
    # Version/wrapper artifacts first, so they can never masquerade as ARCH.
    if "does not support .target" in e or "unsupported .version" in e or \
       "later version" in e or "requires ptx isa .version" in e:
        return "PTXVER"
    # Real arch rejects. ptxas phrases these as
    #   "Instruction 'X' not supported on .target 'sm_YYY'"
    #   "Feature '.z' not supported on .target 'sm_YYY'"
    #   "Instruction 'X' cannot be compiled for architecture 'sm_YYY'"
    #   "requires .target sm_YYY or higher"
    if "not supported on .target" in e or "requires .target" in e or \
       "not supported for .target" in e or "only supported on" in e or \
       "is not supported on" in e or \
       "cannot be compiled for architecture" in e:
        return "ARCH"
    if "not a name of any known instruction" in e:
        return "UNKNOWN-INSTR"
    if "arguments mismatch" in e or "unexpected instruction types" in e or \
       "wrong type" in e or "incorrect instruction type" in e or \
       "operand type" in e:
        return "OPERAND-TYPE"
    if "parsing error" in e or "syntax error" in e:
        return "SYNTAX"
    if "invalid" in e:
        return "INVALID"
    return "OTHER"

def run_ptxas(ptx_path, arch, out_path):
    # PTXAS env var lets us pin an exact toolchain (e.g. 13.2 vs 13.3) by full
    # path; defaults to whatever "ptxas" resolves to on PATH.
    cmd = [os.environ.get("PTXAS", "ptxas"), "-arch=" + arch, ptx_path, "-o", out_path]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, (p.stderr or "") + (p.stdout or "")

def first_err_line(err):
    for line in err.splitlines():
        if "error" in line.lower():
            # strip file path noise
            line = re.sub(r"^ptxas\s+", "", line)
            line = re.sub(r"[A-Za-z]:[\\/][^,]*,", "<file>,", line)
            return line.strip()[:300]
    return err.strip().splitlines()[0][:300] if err.strip() else ""

def probe_one(name, code, extra_decls="", raw=None, arches=None,
              no_std_decls=False, version="9.2"):
    arches = arches or ARCHES
    res = {}
    safe = re.sub(r"[^A-Za-z0-9_.]+", "_", name)[:120]
    for arch in arches:
        target = arch  # .target matches -arch
        # Apply the per-target PTX .version floor before assembling.
        start_ver = version
        if float(start_ver) < float(VERSION_FLOOR.get(arch, "0")):
            start_ver = VERSION_FLOOR[arch]
        used_ver = start_ver
        pfx = os.path.join(PROBEDIR, f"{safe}__{arch}")
        ptx_path = pfx + ".ptx"
        cubin = pfx + ".cubin"

        def _asm(v):
            ptx = build_ptx(code, extra_decls, target=target, version=v,
                            raw=raw, no_std_decls=no_std_decls)
            with open(ptx_path, "w") as f:
                f.write(ptx)
            return run_ptxas(ptx_path, arch, cubin)

        rc, err = _asm(used_ver)
        # If ptxas asks for a newer .version, bump to what it asks for (capped at
        # the max this ptxas accepts). Loop in case the floor rises in steps.
        for _ in range(3):
            if rc == 0:
                break
            m = re.search(r"requires PTX ISA \.version (\d+\.\d+)", err)
            if not m:
                break
            want = m.group(1)
            if float(want) <= float(used_ver) or float(want) > float(MAX_PTX_VERSION):
                break
            used_ver = want
            rc, err = _asm(used_ver)

        if rc == 0:
            res[arch] = {"status": "ACCEPT"}
            if used_ver != version:
                res[arch]["note"] = "needs PTX " + used_ver
        else:
            res[arch] = {"status": "REJECT", "class": classify(err),
                         "err": first_err_line(err)}
            try: os.remove(cubin)
            except OSError: pass
    return res

def main():
    # probes.json lines: {"family":..,"name":..,"code":..,("decls":..)|("raw":..)}
    # usage: runner.py [probes.json] [results.json]
    probes_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "probes.json")
    with open(probes_file) as f:
        probes = json.load(f)
    results_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    for pr in probes:
        key = pr["family"] + "|" + pr["name"]
        if key in results and not pr.get("force"):
            continue
        r = probe_one(pr["name"], pr.get("code", ""), pr.get("decls", ""),
                      raw=pr.get("raw"), no_std_decls=pr.get("no_std_decls", False),
                      version=pr.get("version", "8.8"), arches=pr.get("arches"))
        results[key] = {"family": pr["family"], "name": pr["name"],
                        "code": pr.get("code", "")[:400], "res": r}
        line = key + " :: " + " ".join(
            f"{a}={'OK' if r[a]['status']=='ACCEPT' else 'NO(' + r[a]['class'] + ')'}"
            for a in r)
        print(line, flush=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    main()
