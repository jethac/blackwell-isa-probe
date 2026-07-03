"""Extend an existing results.json with additional target archs.

Pass 1 ran sm_103a/sm_110a over probes.json. Pass 2 (the fix) re-ran every
sm_110a cell that had rejected with the wrapper artifact
"PTX .version 8.8 does not support .target sm_110a" — sm_110 requires PTX ISA
>= 9.0, and runner.probe_one's auto-upgrade only fires on the
"requires PTX ISA .version X.Y" error form, so those cells were version
artifacts, not real ARCH answers. Pass 2 also extended probes_extra.json
(family 8_extra) to the new archs. VERSION_FLOOR encodes the fix.
probes_sink.json (family 9_sink, SASS-dump sinks duplicating main probes) was
deliberately NOT extended; those cells stay unprobed on sm_103a/sm_110a.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner

NEW = ["sm_103a", "sm_110a"]
VERSION_FLOOR = {"sm_110a": 9.0}  # sm_110 does not exist before PTX ISA 9.0
HERE = os.path.dirname(os.path.abspath(__file__))

probes = []
for src in ("probes.json", "probes_extra.json"):
    with open(os.path.join(HERE, src)) as f:
        probes += json.load(f)

rp = os.path.join(HERE, "results.json")
with open(rp) as f:
    results = json.load(f)

n = 0
for pr in probes:
    key = pr["family"] + "|" + pr["name"]
    if key not in results:
        continue
    have = results[key]["res"]
    todo = []
    for a in NEW:
        r = have.get(a)
        if r is None:
            todo.append(a)  # never probed on this arch
        elif r["status"] == "REJECT" and \
                f"does not support .target {a}" in r.get("err", ""):
            todo.append(a)  # wrapper version artifact — re-probe
    if not todo:
        continue
    for a in todo:
        ver = pr.get("version", "8.8")
        if float(ver) < VERSION_FLOOR.get(a, 0):
            ver = str(VERSION_FLOOR[a])
        r = runner.probe_one(pr["name"], pr.get("code", ""), pr.get("decls", ""),
                             raw=pr.get("raw"), no_std_decls=pr.get("no_std_decls", False),
                             version=ver, arches=[a])
        have.update(r)
    n += 1
    if n % 50 == 0:
        print(f"...{n} probes extended", flush=True)

with open(rp, "w") as f:
    json.dump(results, f, indent=1)
print(f"extended/re-probed {n} probes over {NEW}")
