"""Phase 2 report: the range switch against SECOND alone and the Phase 1 R0+T2 and R2+T2 (fit detectors, 6 seeds).

    python results/next_round/phase2_report.py > results/next_round/phase2/report.md
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
P1, P2 = HERE / "phase1", HERE / "phase2"
GROUPS = ("overall", "highway", "urban", "night", "rain", "clear", "adverse")
BINS = ("all", "0-30", "30-50", "50-100", "100-200", "50-200")
METRICS = ("APCD", "APBEV@0.3", "AP3D@0.5", "recall@BEV0.3")


def runs(paths, model):
    out = []
    for p in paths:
        out += json.load(open(p)).get(model, {}).get("runs", [])
    return sorted(out, key=lambda r: r["seed"])


def arr(rs, split, g, b, metric="APCD"):
    return np.array([r[split][g][b][metric] for r in rs])


def main():
    sec = json.load(open(P1 / "detectors_fit/aimotive_second.json"))
    final = [P1 / "final_s123.json", P1 / "final_s456.json"]
    models = {"R0+T2": runs(final, "R0+T2"), "R2+T2": runs(final, "R2+T2"),
              "switch(R0+T2)": runs([P2 / "switch.json"], "switch(R0+T2)"),
              "switch(R2+T2)": runs([P2 / "switch.json"], "switch(R2+T2)")}
    names = list(models)
    for split in ("select", "val"):
        print(f"## {split}: APCD by condition and distance (fusion: mean of 6 seeds; radar gain seed against seed)\n")
        print("| Condition | Distance | Cars | SECOND | " + " | ".join(names) + " | switch(R2+T2) − SECOND | radar gain in the switch |")
        print("|" + "---|" * (6 + len(names)))
        for g in GROUPS:
            for b in BINS:
                e = sec[split].get(g, {}).get(b)
                if not isinstance(e, dict) or not e.get("n_gt"):
                    continue
                cols = [f"{arr(models[m], split, g, b).mean():.3f}" for m in names]
                d = arr(models["switch(R2+T2)"], split, g, b) - arr(models["switch(R0+T2)"], split, g, b)
                print(f"| {g} | {b} | {e['n_gt']:,} | {e['APCD']:.3f} | " + " | ".join(cols) +
                      f" | {arr(models['switch(R2+T2)'], split, g, b).mean() - e['APCD']:+.3f} | "
                      f"{d.mean():+.3f} ({d.min():+.3f} to {d.max():+.3f}, {int((d > 0).sum())} of {len(d)}) |")
        print()
        print(f"### {split}: other metrics, overall beyond 50 m\n")
        print("| Metric | SECOND | " + " | ".join(names) + " |")
        print("|" + "---|" * (2 + len(names)))
        for metric in METRICS:
            print(f"| {metric} | {sec[split]['overall']['50-200'][metric]:.3f} | " +
                  " | ".join(f"{arr(models[m], split, 'overall', '50-200', metric).mean():.3f}" for m in names) + " |")
        nms = {m: [r[split]["far_nms_differences"] for r in models[m]] for m in names if m.startswith("switch")}
        print(f"\nCandidates beyond 50 m kept or dropped by NMS differently from SECOND alone, per seed: {nms}\n")


if __name__ == "__main__":
    main()
