"""The Phase 4 rule of results/next_round/PLAN.md, on select, seeds 1-3, APCD at 50-200 m and 100-200 m.

    python results/next_round/phase4_decide.py     # prints the final configuration and writes phase4/decision.json
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
P4 = HERE / "phase4"
BINS = ("50-200", "100-200")
REF = "R2+T2+K40"
CANDIDATES = ("R2+T2+K40+G", "R2+T2+K40+G+C")
FILES = {"normal": {REF: HERE / "phase3/stageA.json", "switch(R2+T2)": HERE / "phase2/switch.json", **{c: P4 / "candidates.json" for c in CANDIDATES}},
         "nocam": {REF: P4 / "nocam_ref.json", **{c: P4 / "nocam.json" for c in CANDIDATES}}}


def apcd(name, b, kind="normal"):
    rs = [r for r in json.load(open(FILES[kind][name]))[name]["runs"] if r["seed"] in (1, 2, 3)]
    assert len(rs) == 3, (name, kind, len(rs))
    return np.array([r["select"]["overall"][b]["APCD"] for r in sorted(rs, key=lambda r: r["seed"])])


def compare(new, old, kind="normal"):
    return {b: dict(mean=apcd(new, b, kind).mean(), sd=apcd(new, b, kind).std(ddof=1), old=apcd(old, b, kind).mean(),
                    margin=apcd(new, b, kind).mean() - apcd(old, b, kind).mean()) for b in BINS}


def main():
    out = {}
    for c in CANDIDATES:
        beats = compare(c, REF)
        nocam = compare(c, REF, "nocam")
        passes = all(v["margin"] > v["sd"] for v in beats.values())
        vetoed = passes and any(v["margin"] < -v["sd"] for v in nocam.values())
        switch = compare(c, "switch(R2+T2)")
        out[c] = dict(beats_reference=passes, vetoed=vetoed, adopted=passes and not vetoed,
                      beats_switch=all(v["margin"] > v["sd"] for v in switch.values()), reference=beats, nocam=nocam, switch=switch)
    adopted = [c for c in CANDIDATES if out[c]["adopted"]]
    final = max(adopted, key=lambda c: out[c]["reference"]["50-200"]["mean"]) if adopted else REF
    json.dump(dict(final=final, candidates=out), open(P4 / "decision.json", "w"), indent=1)
    print(final)


if __name__ == "__main__":
    main()
