"""The Phase 3 rule of results/next_round/PLAN.md, on select, seeds 1-3, APCD at 50-200 m and 100-200 m.

    python results/next_round/phase3_decide.py stageA            # prints the kept tokens, largest 50-200 m gain first
    python results/next_round/phase3_decide.py beats NEW OLD     # exit 0 if NEW beats OLD
    python results/next_round/phase3_decide.py switch NAME       # exit 0 if NAME beats switch(R2+T2)
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
P3 = HERE / "phase3"
BINS = ("50-200", "100-200")
TOKENS = ("H", "P", "I1", "I2", "K40", "D2", "D3", "S50", "S100")
GROUPS = ({"I1", "I2", "K40"}, {"D2", "D3"}, {"S50", "S100"})
REF = "R2+T2"


def runs(name):
    """Seeds 1-3 of a configuration from any Phase 3 file (or switch(R2+T2) from Phase 2)."""
    files = [HERE / "phase2/switch.json"] if name.startswith("switch") else sorted(P3.glob("stage*.json"))
    for p in files:
        d = json.load(open(p))
        if name in d:
            rs = [r for r in d[name]["runs"] if r["seed"] in (1, 2, 3)]
            assert len(rs) == 3, f"{name}: {len(rs)} seeds in {p}"
            return sorted(rs, key=lambda r: r["seed"])
    raise KeyError(name)


def apcd(name, b):
    return np.array([r["select"]["overall"][b]["APCD"] for r in runs(name)])


def beats(new, old):
    """New's mean is above old's by more than the SD of new's seeds, in both bins."""
    out = {}
    for b in BINS:
        a, o = apcd(new, b), apcd(old, b)
        out[b] = dict(mean=a.mean(), sd=a.std(ddof=1), old=o.mean(), margin=a.mean() - o.mean())
    return all(v["margin"] > v["sd"] for v in out.values()), out


def stage_a():
    passing = {t: beats(f"{REF}+{t}", REF) for t in TOKENS}
    kept = [t for t, (ok, _) in passing.items() if ok]
    for g in GROUPS:
        inside = [t for t in kept if t in g]
        if len(inside) > 1:
            best = max(inside, key=lambda t: passing[t][1]["50-200"]["mean"])
            kept = [t for t in kept if t not in g or t == best]
    kept.sort(key=lambda t: -passing[t][1]["50-200"]["margin"])
    json.dump(dict(kept=kept, tests={t: dict(passes=ok, bins=v) for t, (ok, v) in passing.items()}),
              open(P3 / "decision_A.json", "w"), indent=1)
    return kept


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stageA":
        print(" ".join(stage_a()))
    elif cmd == "beats":
        ok, v = beats(sys.argv[2], sys.argv[3])
        print(sys.argv[2], "beats" if ok else "does not beat", sys.argv[3], json.dumps(v), file=sys.stderr)
        sys.exit(0 if ok else 1)
    elif cmd == "switch":
        ok, v = beats(sys.argv[2], "switch(R2+T2)")
        print(sys.argv[2], "beats" if ok else "does not beat", "the switch", json.dumps(v), file=sys.stderr)
        sys.exit(0 if ok else 1)
