"""The Phase 6 rule of results/next_round/PLAN.md (the Phase 4 rule unchanged), on select, seeds 1-3, APCD at 50-200 m and 100-200 m.

    python results/next_round/phase6_decide.py     # prints the current model and writes phase6/decision.json
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
P4, P6 = HERE / "phase4", HERE / "phase6"
BINS = ("50-200", "100-200")
REF = "R2+T2+K40+G"
CAND = "R2+T2+K40+G+RD"
FILES = {"normal": {REF: P4 / "candidates.json", "switch(R2+T2)": HERE / "phase2/switch.json", CAND: P6 / "candidates.json"},
         "nocam": {REF: P4 / "nocam.json", CAND: P6 / "nocam.json"}}


def apcd(name, b, kind="normal"):
    rs = [r for r in json.load(open(FILES[kind][name]))[name]["runs"] if r["seed"] in (1, 2, 3)]
    assert len(rs) == 3, (name, kind, len(rs))
    return np.array([r["select"]["overall"][b]["APCD"] for r in sorted(rs, key=lambda r: r["seed"])])


def compare(new, old, kind="normal"):
    return {b: dict(mean=apcd(new, b, kind).mean(), sd=apcd(new, b, kind).std(ddof=1), old=apcd(old, b, kind).mean(),
                    margin=apcd(new, b, kind).mean() - apcd(old, b, kind).mean()) for b in BINS}


def main():
    beats = compare(CAND, REF)
    nocam = compare(CAND, REF, "nocam")
    passes = all(v["margin"] > v["sd"] for v in beats.values())
    vetoed = passes and any(v["margin"] < -v["sd"] for v in nocam.values())
    switch = compare(CAND, "switch(R2+T2)")
    out = dict(beats_reference=passes, vetoed=vetoed, adopted=passes and not vetoed,
               beats_switch=all(v["margin"] > v["sd"] for v in switch.values()), reference=beats, nocam=nocam, switch=switch)
    current = CAND if out["adopted"] else REF
    json.dump(dict(current=current, candidates={CAND: out}), open(P6 / "decision.json", "w"), indent=1)
    print(current)


if __name__ == "__main__":
    main()
