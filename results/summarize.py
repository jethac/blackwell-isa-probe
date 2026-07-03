#!/usr/bin/env python3
"""Regenerate the accept/reject matrix, the cross-target deltas, and the camp
summary straight from the raw probe data.

Primary source is results_full.json -- the comprehensive sweep (the curated
serving-kernel families plus a systematic pass over the documented PTX
instruction set), one internally-consistent run at one ptxas version. If that
file is absent it falls back to the original curated results.json +
gap_results.json (the 296-probe companion sweep).

Nothing here is hand-typed: every cell is ptxas's own verdict as recorded by the
harness. Run it to check the README's tables and headline numbers against the
source of truth.

    python results/summarize.py            # from the repo root
"""
import json, os
from collections import OrderedDict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_120f", "sm_121a"]
CHIP = {"sm_100a": "B200", "sm_103a": "B300", "sm_110a": "Thor",
        "sm_120a": "RTX50", "sm_120f": "120f", "sm_121a": "Spark"}
TCGEN05_CAMP = {"sm_100a", "sm_103a", "sm_110a"}   # B200, B300, Thor
WARP_BS_CAMP = {"sm_120a", "sm_120f", "sm_121a"}   # RTX 50, 120f family, Spark
# The curated serving-kernel families (tensor / memory / sync surface) vs the
# comprehensive pass over the rest of the documented PTX instruction set.
CURATED_FAMS = {"1_blockscale_mma", "2_plain_mma", "2_sparse_mma", "3_cvt",
                "4_ldstmatrix", "5_async", "6_cluster", "7_misc",
                "G1_simd", "G3_conv", "25_tensor_extra"}


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def cell(res, arch):
    r = res.get(arch)
    if r is None:
        return "."                       # not probed on this target
    return "Y" if r["status"] == "ACCEPT" else "N"


def pat(v):
    return "".join(cell(v["res"], c) for c in COLS)


def print_matrix(data, title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    hdr = "%-46s %s" % ("probe", " ".join("%-5s" % c.replace("sm_", "") for c in COLS))
    fam = None
    for key, v in data.items():
        if v["family"] != fam:
            fam = v["family"]
            print(f"\n-- family {fam} --")
            print(hdr)
        print("%-46s %s" % (v["name"][:46],
                            " ".join("%-5s" % cell(v["res"], c) for c in COLS)))


def deltas(data):
    """Every probe whose verdict is not identical across all six targets."""
    print(f"\n{'=' * 78}\nCROSS-TARGET DELTAS (verdict differs across the six targets)\n{'=' * 78}")
    groups = OrderedDict()
    for v in data.values():
        p = pat(v)
        if "." in p:
            continue
        if p.count("Y") in (0, 6):        # universal accept / universal reject
            continue
        groups.setdefault(p, []).append((v["family"], v["name"]))
    labels = {
        "NNNYYY": "warp-block-scale camp only (RTX50 / 120f / Spark)",
        "YYYNNN": "tcgen05 camp only (B200 / B300 / Thor)",
        "YYNNNN": "B200 / B300 only (not Thor, not consumer)",
        "NNNYNY": "sm_120a & sm_121a only (a-suffix; not the 120f family target)",
    }
    for p in sorted(groups, key=lambda x: -len(groups[x])):
        lab = labels.get(p, "")
        items = groups[p]
        new = [i for i in items if i[0] not in CURATED_FAMS]
        print(f"\n  pattern {p}  ({len(items)} probes)  {lab}")
        if new:
            print(f"    ^ {len(new)} of these are in the comprehensive (non-curated) families:")
            for fam, name in new:
                print(f"        [{fam}] {name}")
    return groups


def camp_summary(data):
    print(f"\n{'=' * 78}\nCAMP SUMMARY (Y=ptxas accepts, N=arch-reject)\n{'=' * 78}")
    for a, b, lbl in [("sm_120a", "sm_121a", "Spark(121a) vs 5090(120a)"),
                      ("sm_120a", "sm_120f", "5090(120a) vs family(120f)"),
                      ("sm_100a", "sm_103a", "B200(100a) vs B300(103a)"),
                      ("sm_110a", "sm_100a", "Thor(110a) vs B200(100a)")]:
        co = diff = 0
        names = []
        for v in data.values():
            x, y = v["res"].get(a), v["res"].get(b)
            if x and y:
                co += 1
                if x["status"] != y["status"]:
                    diff += 1
                    names.append(v["name"])
        print(f"  {lbl}: {co} co-probed, {diff} differences")
        for n in names:
            print(f"      {n}")


def coverage(data):
    print(f"\n{'=' * 78}\nCOVERAGE\n{'=' * 78}")
    fams = Counter(v["family"] for v in data.values())
    curated = sum(n for f, n in fams.items() if f in CURATED_FAMS)
    comp = len(data) - curated
    patc = Counter(pat(v) for v in data.values())
    universal = patc.get("YYYYYY", 0)
    print(f"  {len(data)} probes total: {curated} curated (tensor/memory/sync) "
          f"+ {comp} comprehensive (rest of PTX)")
    print(f"  {len(data) * len(COLS)} ptxas invocations across {len(COLS)} targets")
    print(f"  accepted on all six targets (universal Blackwell baseline): {universal}")
    print(f"  rejected on all six (illegal form / absent on every target): "
          f"{patc.get('NNNNNN', 0)}")
    split = sum(n for p, n in patc.items()
                if "." not in p and p.count("Y") not in (0, 6))
    print(f"  cross-target splits (a real per-target ISA difference): {split}")
    # per-family split count
    fam_split = Counter()
    for v in data.values():
        p = pat(v)
        if "." not in p and p.count("Y") not in (0, 6):
            fam_split[v["family"]] += 1
    print("  splits by family:")
    for f in sorted(fam_split):
        tag = "curated" if f in CURATED_FAMS else "COMPREHENSIVE"
        print(f"      {fam_split[f]:3d}  {f}  ({tag})")


def main():
    full = os.path.join(HERE, "results_full.json")
    if os.path.exists(full):
        data = load("results_full.json")
        print(f"comprehensive sweep: {len(data)} probes x {len(COLS)} targets "
              f"(results_full.json)")
    else:
        m, g = load("results.json"), load("gap_results.json")
        data = OrderedDict(m); data.update(g)
        print(f"curated sweep: {len(m)} main + {len(g)} gap probes "
              f"(results.json + gap_results.json; results_full.json not found)")
    print("columns: 100a(B200) 103a(B300) 110a(Thor) | 120a(RTX50) 120f(family) 121a(Spark)")
    print_matrix(data, "FULL MATRIX")
    deltas(data)
    camp_summary(data)
    coverage(data)


if __name__ == "__main__":
    main()
