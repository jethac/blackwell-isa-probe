#!/usr/bin/env python3
"""ISA compile-probe harness for consumer Blackwell audit.

Wraps instruction snippets in a minimal .ptx file and runs ptxas for each
target arch. Records ACCEPT / REJECT(class) per arch. ptxas is the oracle.
"""
import json, os, re, subprocess, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
PROBEDIR = os.path.join(HERE, "probes")
os.makedirs(PROBEDIR, exist_ok=True)

ARCHES = ["sm_120a", "sm_121a", "sm_120f", "sm_100a"]

STD_DECLS = """\
    .reg .f32 f<16>;
    .reg .b32 r<16>;
    .reg .b32 a<8>;
    .reg .b32 b<8>;
    .reg .b32 sfa;
    .reg .b32 sfb;
    .reg .b32 e0;
    .reg .b64 rd<8>;
    .reg .pred p<4>;
    .reg .b16 h<16>;
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
    e = err.lower()
    if "not supported on .target" in e or "requires .target" in e or \
       "not supported for .target" in e or "only supported on" in e or \
       "is not supported on" in e:
        return "ARCH"
    if "not a name of any known instruction" in e:
        return "UNKNOWN-INSTR"
    if "unexpected instruction types" in e or "wrong type" in e or \
       "incorrect instruction type" in e:
        return "OPERAND-TYPE"
    if "parsing error" in e or "syntax error" in e:
        return "SYNTAX"
    if "unsupported .version" in e or "later version" in e:
        return "PTXVER"
    if "invalid" in e:
        return "INVALID"
    return "OTHER"

def run_ptxas(ptx_path, arch, out_path):
    cmd = ["ptxas", "-arch=" + arch, ptx_path, "-o", out_path]
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
              no_std_decls=False, version="8.8"):
    arches = arches or ARCHES
    res = {}
    safe = re.sub(r"[^A-Za-z0-9_.]+", "_", name)[:120]
    for arch in arches:
        target = arch  # .target matches -arch
        ptx = build_ptx(code, extra_decls, target=target, version=version,
                        raw=raw, no_std_decls=no_std_decls)
        pfx = os.path.join(PROBEDIR, f"{safe}__{arch}")
        ptx_path = pfx + ".ptx"
        cubin = pfx + ".cubin"
        with open(ptx_path, "w") as f:
            f.write(ptx)
        rc, err = run_ptxas(ptx_path, arch, cubin)
        used_ver = version
        m = re.search(r"requires PTX ISA \.version (\d+\.\d+)", err) if rc else None
        if rc != 0 and m:
            used_ver = m.group(1)
            ptx = build_ptx(code, extra_decls, target=target, version=used_ver,
                            raw=raw, no_std_decls=no_std_decls)
            with open(ptx_path, "w") as f:
                f.write(ptx)
            rc, err = run_ptxas(ptx_path, arch, cubin)
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
    probes_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "probes.json")
    with open(probes_file) as f:
        probes = json.load(f)
    results_path = os.path.join(HERE, "results.json")
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
