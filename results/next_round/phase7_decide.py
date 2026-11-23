"""The pre-registered choices of Phase 7 (results/next_round/PLAN.md), all on select.

    python results/next_round/phase7_decide.py radar_input    # d1 or d2: RadarDistill's radar points (higher detector-alone APCD 50-200)
    python results/next_round/phase7_decide.py detector       # dets dir of the best of D1, D2, D3 (detector-alone APCD 50-200)
    python results/next_round/phase7_decide.py radar_expert   # "<config> <weights dir>": LRf or LRd, higher mean APCD 50-200
    python results/next_round/phase7_decide.py final          # the current-model rule; writes phase7/decision.json
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
P7 = HERE / "phase7"
DETECTORS = {"d1": (HERE / "phase6/radar_detector.json", "runs/next_round/phase6/radar_dets"),
             "d2": (P7 / "radar_detector_d2.json", "runs/next_round/phase7/d2_dets"),
             "d3": (P7 / "radar_detector_d3.json", "runs/next_round/phase7/d3_dets")}
EXPERTS = {"R2+T2+NC": (P7 / "experts.json", "runs/next_round/phase7/fusion"),
           "R0+T2+RP": (P7 / "experts_rp.json", "runs/next_round/phase7/fusion_rp")}
INCUMBENT = ("R2+T2+K40+G", [HERE / "phase4/final_s123.json", HERE / "phase4/final_s456.json"])


def detector_apcd(tag):
    path = DETECTORS[tag][0]
    return json.load(open(path))["select"]["overall"]["50-200"]["APCD"] if path.exists() else None


def runs(name, files):
    out = []
    for f in files:
        if pathlib.Path(f).exists():
            out += json.load(open(f)).get(name, {}).get("runs", [])
    return sorted({r["seed"]: r for r in out}.values(), key=lambda r: r["seed"])


def apcd(rs, b):
    return np.array([r["select"]["overall"][b]["APCD"] for r in rs])


def main():
    what = sys.argv[1]
    if what == "radar_input":
        a = {t: detector_apcd(t) for t in ("d1", "d2")}
        assert all(v is not None for v in a.values()), a
        print(max(a, key=a.get))
    elif what == "detector":
        a = {t: detector_apcd(t) for t in DETECTORS}
        a = {t: v for t, v in a.items() if v is not None}
        print(DETECTORS[max(a, key=a.get)][1])
    elif what == "radar_expert":
        m = {n: apcd(runs(n, [f]), "50-200").mean() for n, (f, _) in EXPERTS.items()}
        best = max(m, key=m.get)
        print(best, EXPERTS[best][1])
    elif what == "final":
        inc = runs(*INCUMBENT)
        cands = {n: runs(n, [f]) for n, (f, _) in EXPERTS.items()}
        sel = json.load(open(P7 / "selector.json")) if (P7 / "selector.json").exists() else {}
        cands.update({k: sorted(v["runs"], key=lambda r: r["seed"]) for k, v in sel.items()})
        cands["R0+T2+NC"] = runs("R0+T2+NC", [P7 / "experts.json"])
        out = {}
        for n, rs in cands.items():
            if len(rs) != 6:
                continue
            res = {b: dict(mean=float(apcd(rs, b).mean()), sd=float(apcd(rs, b).std(ddof=1)), incumbent=float(apcd(inc, b).mean()))
                   for b in ("50-200", "100-200")}
            out[n] = dict(res, beats=all(v["mean"] - v["incumbent"] > v["sd"] for v in res.values()))
        winners = [n for n, v in out.items() if v["beats"]]
        current = max(winners, key=lambda n: out[n]["50-200"]["mean"]) if winners else INCUMBENT[0]
        json.dump(dict(current=current, candidates=out), open(P7 / "decision.json", "w"), indent=1)
        print(current)


if __name__ == "__main__":
    main()
