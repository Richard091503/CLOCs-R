"""Phase 1 rule of PLAN.md (addendum): cross-fitted or in-sample fusion training inputs, from select only.

    python results/next_round/phase1_decide.py     # prints the chosen arm, writes results/next_round/phase1/decision.json
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent / "phase1"


def far(path, model):
    return np.array([r["select"]["overall"]["50-200"]["APCD"] for r in json.load(open(path))[model]["runs"]])


def main():
    out = {}
    for m in ("R2+T2", "R0+T2"):
        a, b = far(HERE / "select_insample.json", m), far(HERE / "select_crossfit.json", m)
        out[m] = dict(insample=a.tolist(), crossfit=b.tolist(), diff=float(b.mean() - a.mean()), sd_crossfit=float(b.std(ddof=1)))
    r = out["R2+T2"]
    out["arm"] = "crossfit" if r["diff"] > r["sd_crossfit"] else "insample"
    json.dump(out, open(HERE / "decision.json", "w"), indent=1)
    print(out["arm"])


if __name__ == "__main__":
    main()
