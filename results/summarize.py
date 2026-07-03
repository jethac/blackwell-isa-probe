#!/usr/bin/env python3
"""Regenerate the accept/reject matrix straight from the raw probe data.

Reads results.json (the 296-probe main sweep) and gap_results.json (the 39
gap-fill probes) and prints a per-family Y/N/. table plus the camp-membership
summary the README quotes. Nothing here is hand-typed: every cell comes from
ptxas's own verdict as recorded by the harness. Run it to check the README
table against the source of truth.

    python results/summarize.py            # from the repo root
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_120f", "sm_121a"]
TCGEN05_CAMP = {"sm_100a", "sm_103a", "sm_110a"}      # B200, B300, Thor
WARP_BS_CAMP = {"sm_120a", "sm_120f", "sm_121a"}      # RTX 50, 120f family, Spark


def load(name):
    p = os.path.join(HERE, name)
    with open(p) as f:
        return json.load(f)


def cell(res, arch):
    r = res.get(arch)
    if r is None:
        return "."                       # not probed on this target
    return "Y" if r["status"] == "ACCEPT" else "N"


def print_matrix(data, title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    hdr = "%-46s %s" % ("probe", " ".join("%-5s" % c.replace("sm_", "") for c in COLS))
    fam = None
    for key, v in data.items():
        if v["family"] != fam:
            fam = v["family"]
            print(f"\n-- family {fam} --")
            print(hdr)
        row = " ".join("%-5s" % cell(v["res"], c) for c in COLS)
        print("%-46s %s" % (v["name"][:46], row))


def camp_summary(main, gap):
    print(f"\n{'='*78}\nCAMP SUMMARY (Y=ptxas accepts, N=arch-reject)\n{'='*78}")
    alldata = dict(main); alldata.update(gap)

    # Spark (121a) vs 5090 (120a): count co-probed cells and differences.
    co = diff = 0
    for v in alldata.values():
        a, b = v["res"].get("sm_120a"), v["res"].get("sm_121a")
        if a and b:
            co += 1
            if a["status"] != b["status"]:
                diff += 1
                print("  120a!=121a:", v["name"])
    print(f"Spark(sm_121a) vs 5090(sm_120a): {co} co-probed, {diff} differences")

    # 5090 (120a) vs 12.x family target (120f).
    co = diff = 0
    for v in alldata.values():
        a, b = v["res"].get("sm_120a"), v["res"].get("sm_120f")
        if a and b:
            co += 1
            if a["status"] != b["status"]:
                diff += 1
                print("  120a!=120f:", v["name"], a["status"], "vs", b["status"])
    print(f"5090(sm_120a) vs family(sm_120f): {co} co-probed, {diff} differences")

    # B200 (100a) vs B300 (103a).
    co = diff = 0
    for v in alldata.values():
        a, b = v["res"].get("sm_100a"), v["res"].get("sm_103a")
        if a and b:
            co += 1
            if a["status"] != b["status"]:
                diff += 1
                print("  100a!=103a:", v["name"])
    print(f"B200(sm_100a) vs B300(sm_103a): {co} co-probed, {diff} differences")


def main():
    m = load("results.json")
    g = load("gap_results.json")
    print(f"main sweep: {len(m)} probes | gap-fill: {len(g)} probes")
    print("columns: 100a(B200) 103a(B300) 110a(Thor) | 120a(RTX50) 120f(family) 121a(Spark)")
    print_matrix(m, "MAIN SWEEP (results.json)")
    print_matrix(g, "GAP-FILL (gap_results.json)")
    camp_summary(m, g)


if __name__ == "__main__":
    main()
