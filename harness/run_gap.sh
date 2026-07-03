#!/usr/bin/env bash
# ISA gap-fill orchestrator. Run inside the CUDA 13.2 devel container.
# Closes footnote-7 (u8x4/s8x4 SIMD) and footnote-8 (FMNMX3) gaps.
#
# NOTE (repo layout): preserved as an as-run record. It was written for a FLAT
# working directory where the gap generators (gen_gap_probes.py, gen_sinks.py),
# the harness (runner.py) and the SASS probe (fmnmx3.cu) all sit next to this
# script. In this repo those files live under generators/, harness/ and probes/.
# To reproduce, copy them into one directory alongside this script (or run the
# commands from the README's "How to run" section by hand).
set -u
cd "$(dirname "$0")"
ARCHES6="sm_100a sm_103a sm_110a sm_120a sm_120f sm_121a"
ARCHES7="sm_90a $ARCHES6"

banner(){ echo; echo "==================== $* ===================="; }

banner "BOX + TOOLCHAIN VERSIONS"
echo "--- uname ---";        uname -a
echo "--- nvidia-smi ---";   nvidia-smi --query-gpu=name,driver_version --format=csv 2>&1 | head -3 || echo "no gpu query"
echo "--- ptxas --version ---";     ptxas --version
echo "--- cuobjdump --version ---"; cuobjdump --version
echo "--- nvdisasm --version ---";  nvdisasm --version
echo "--- nvcc --version ---";      nvcc --version 2>&1 | tail -4

banner "GAP 1 - u8x4/s8x4 SIMD int ops (footnote 7): ACCEPT/REJECT sweep, all 6 targets"
python3 gen_gap_probes.py
python3 runner.py probes_gap.json
echo
echo "--- ACCEPT/REJECT matrix (columns: 100a 103a 110a 120a 120f 121a) ---"
python3 - <<'PY'
import json
r=json.load(open("results.json"))
cols=["sm_100a","sm_103a","sm_110a","sm_120a","sm_120f","sm_121a"]
def cell(d):
    if d is None: return "  .   "
    if d["status"]=="ACCEPT": return " ACC* " if d.get("note") else "  ACC "
    return "no:"+d.get("class","?")[:4].ljust(3)
print("%-24s %s"%("probe"," ".join("%-6s"%c.replace('sm_','') for c in cols)))
for k,v in r.items():
    res=v["res"]
    print("%-24s %s"%(v["name"], " ".join("%-6s"%cell(res.get(c)) for c in cols)))
print("\n* = accepted but ptxas demanded a higher PTX .version (per-cell note in results.json)\n")
docset=[f"{op}.{t}" for op in("add","sub","min","max") for t in("u8x4","s8x4")]+["neg.s8x4"]
print("DOCUMENTED SET verdict (add/sub/min/max on u8x4/s8x4, neg.s8x4):")
for name in docset:
    v=r.get("G1_simd|"+name)
    if not v: continue
    res=v["res"]
    acc=[c.replace("sm_","") for c in cols if res.get(c,{}).get("status")=="ACCEPT"]
    rej=[c.replace("sm_","") for c in cols if res.get(c,{}).get("status")=="REJECT"]
    print("  %-10s accept=[%s]  reject=[%s]"%(name, ",".join(acc), ",".join(rej)))
# one representative reject error text, to show it's an ARCH gate not a syntax miss
rep=r.get("G1_simd|min.u8x4",{}).get("res",{}).get("sm_100a",{})
print("\n  sm_100a reject error for min.u8x4:", rep.get("err","<n/a>"))
PY

banner "GAP 1 - SASS confirmation on sm_120a (real HW SIMD instr vs PRMT/emulation)"
python3 gen_sinks.py sm_120a
for f in sinks/sink_*.ptx; do
    op=$(basename "$f" .ptx)
    if ptxas -arch=sm_120a -O3 "$f" -o "sinks/${op}.cubin" 2>"sinks/${op}.err"; then
        echo "----- $op  (sm_120a -O3, non-boilerplate SASS) -----"
        cuobjdump -sass "sinks/${op}.cubin" | grep -oE '/\*[0-9a-f]+\*/ +@?!?P?[0-9]? *[A-Z][A-Z0-9_.]+' | \
            grep -vE '\b(MOV|IMAD|ULDC|EXIT|BRA|NOP|S2R|CS2R|RET)\b' || echo "   (only boilerplate?)"
    else
        echo "----- $op : ptxas REJECT on sm_120a -----"; head -2 "sinks/${op}.err"
    fi
done
echo
echo "--- cross-check: min.u8x4 sink must REJECT on sm_100a (datacenter) ---"
sed 's/sm_120a/sm_100a/' sinks/sink_min_u8x4.ptx > sinks/sink_min_u8x4_100a.ptx
ptxas -arch=sm_100a -O3 sinks/sink_min_u8x4_100a.ptx -o /tmp/x100.cubin 2>&1 | head -3 && echo "(built - UNEXPECTED)" || echo "(rejected as expected)"

banner "GAP 2 - FMNMX3 / VIMNMX3 (footnote 8): chained min/max SASS at -O3, per arch"
for A in $ARCHES7; do
    if nvcc -arch=$A --cubin -Xptxas -O3 fmnmx3.cu -o fmnmx3_$A.cubin 2>nvcc_$A.err; then
        cuobjdump -sass fmnmx3_$A.cubin > sass_$A.txt 2>&1
        echo "   $A : built OK"
    else
        echo "   $A : nvcc could NOT build -> $(head -1 nvcc_$A.err)"
        # fallback: ptxas-from-PTX at this target for the family/exotic arches
        if nvcc -arch=$A --ptx fmnmx3.cu -o fmnmx3_$A.ptx 2>>nvcc_$A.err && \
           ptxas -arch=$A -O3 fmnmx3_$A.ptx -o fmnmx3_$A.cubin 2>>nvcc_$A.err; then
            cuobjdump -sass fmnmx3_$A.cubin > sass_$A.txt 2>&1
            echo "        (recovered via nvcc --ptx + ptxas -O3)"
        fi
    fi
done
echo
echo "--- FMNMX3 full-row tally (word-boundary mnemonic counts across f3/i3/u3) ---"
python3 - <<'PY'
import re,glob,os
mnem=["FMNMX3","VIMNMX3","IMNMX3","FMNMX","VIMNMX","IMNMX"]
order=["sm_90a","sm_100a","sm_103a","sm_110a","sm_120a","sm_120f","sm_121a"]
print("%-9s %-8s %-8s %-8s %-8s %-8s %-8s   verdict"%(("arch",)+tuple(mnem)))
for A in order:
    fn="sass_%s.txt"%A
    if not os.path.exists(fn):
        print("%-9s  (not built)"%A); continue
    t=open(fn).read()
    c={m:len(re.findall(r'\b'+m+r'\b', t)) for m in mnem}
    f3in = c["FMNMX3"]>0
    i3in = (c["VIMNMX3"]>0 or c["IMNMX3"]>0)
    verdict = "float3=%s int3=%s"%("Y" if f3in else "n", "Y" if i3in else "n")
    print("%-9s %-8d %-8d %-8d %-8d %-8d %-8d   %s"%(A,c["FMNMX3"],c["VIMNMX3"],c["IMNMX3"],c["FMNMX"],c["VIMNMX"],c["IMNMX"],verdict))
print("\nfloat3=Y  -> HW fuses fmax(fmax(a,b),c) into ONE FMNMX3")
print("float3=n  -> two FMNMX (2-input); softmax running-max pays one extra instr/step")
print("int3=Y    -> HW fuses max(max(a,b),c) into ONE VIMNMX3/IMNMX3")
PY

banner "GAP 2 - raw SASS snippet: MNMX lines, 90a / 100a / 120a / 120f / 121a"
for A in sm_90a sm_100a sm_120a sm_120f sm_121a; do
    [ -f sass_$A.txt ] || continue
    echo "----- $A -----"
    grep -nE '\b(FMNMX3|FMNMX|VIMNMX3|IMNMX3|IMNMX|VIMNMX)\b' sass_$A.txt | head -14
done

banner "DONE"
