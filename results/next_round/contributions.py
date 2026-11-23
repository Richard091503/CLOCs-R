"""What each addition contributes, from the stored results (no new runs): APCD differences between configurations that
differ by one addition, seed against seed where both have seeds 1-6, on select and val, by distance bin.

    python results/next_round/contributions.py > results/next_round/contributions.md
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
BINS = ("all", "0-30", "30-50", "50-100", "100-200", "50-200")
FILES = {
    "R0+T2": ["phase1/final_s123.json", "phase1/final_s456.json"],
    "R2+T2": ["phase1/final_s123.json", "phase1/final_s456.json"],
    "switch(R2+T2)": ["phase2/switch.json"],
    "switch(R0+T2)": ["phase2/switch.json"],
    "R2+T2+K40": ["phase3/final_s123.json", "phase3/final_s456.json"],
    "R0+T2+K40": ["phase3/final_r0_s123.json", "phase3/final_s456.json"],
    "R2+T2+K40+G": ["phase4/final_s123.json", "phase4/final_s456.json"],
    "R0+T2+K40+G": ["phase4/final_r0_s123.json", "phase4/final_s456.json"],
    "R2+T2+K40+G+RD": ["phase6/final_s123.json", "phase6/final_s456.json"],
    "ablations": ["contributions_runs.json"],  # filled in later by the missing-ablation runs, if present
}
# (addition, configuration with it, configuration without it)
STEPS = [
    ("Camera-LiDAR fusion network (vs SECOND alone)", "R0+T2", "SECOND"),
    ("Residual scoring T2 (vs plain CLOCs output), no radar", "R0+T2", "R0"),
    ("Far-range loss weight 2 (vs 1), no radar", "R0+T2", "R0+T2@fw1"),
    ("Radar Doppler feature R2, plain fusion", "R2+T2", "R0+T2"),
    ("Radar Doppler feature R2, with K40", "R2+T2+K40", "R0+T2+K40"),
    ("Radar Doppler feature R2, gated model", "R2+T2+K40+G", "R0+T2+K40+G"),
    ("Top-40 2D boxes K40", "R2+T2+K40", "R2+T2"),
    ("Gated model G", "R2+T2+K40+G", "R2+T2+K40"),
    ("Range switch at 50 m (vs R2+T2)", "switch(R2+T2)", "R2+T2"),
    ("Range switch at 50 m (vs SECOND)", "switch(R2+T2)", "SECOND"),
    ("Radar-detector candidates RD", "R2+T2+K40+G+RD", "R2+T2+K40+G"),
    ("Camera in the gated model (vs camera removed)", "R2+T2+K40+G", "R2+T2+K40+G@nocam"),
]


def runs(name):
    if name == "SECOND":
        sec = json.load(open(HERE / "phase1/detectors_fit/aimotive_second.json"))
        return [dict(seed=0, **{s: sec[s] for s in ("select", "val")})]
    out = []
    for f in FILES.get(name, []) + FILES["ablations"]:
        p = HERE / f
        if p.exists():
            out += json.load(open(p)).get(name, {}).get("runs", [])
    return sorted({r["seed"]: r for r in out}.values(), key=lambda r: r["seed"])


def diff(a, b, split, bin_):
    ra, rb = runs(a), runs(b)
    if not ra or not rb or split not in ra[0] or split not in rb[0]:
        return None
    va = {r["seed"]: r[split]["overall"][bin_]["APCD"] for r in ra}
    vb = {r["seed"]: r[split]["overall"][bin_]["APCD"] for r in rb}
    if set(vb) == {0}:  # SECOND: one deterministic run
        d = np.array([v - vb[0] for v in va.values()])
    else:
        common = sorted(set(va) & set(vb))
        if not common:
            return None
        d = np.array([va[s] - vb[s] for s in common])
    return d


def main():
    print("# What each addition contributes (APCD; mean difference, seed against seed, positive seeds)\n")
    print("Each row compares two configurations that differ by one addition. \"-\" means that comparison has not been run.\n")
    for split in ("val", "select"):
        print(f"## {split}\n")
        print("| Addition | With | Without | " + " | ".join(BINS) + " |")
        print("|" + "---|" * (3 + len(BINS)))
        for label, a, b in STEPS:
            cells = []
            for bin_ in BINS:
                d = diff(a, b, split, bin_)
                cells.append("-" if d is None else f"{d.mean():+.3f} ({int((d > 0).sum())}/{len(d)})")
            print(f"| {label} | {a} | {b} | " + " | ".join(cells) + " |")
        print()
    print("## Absolute APCD, overall (mean of seeds)\n")
    print("| Configuration | val all | val 30-50 | val 50-200 | select all | select 50-200 |")
    print("|---|---|---|---|---|---|")
    for name in ["SECOND", "R0+T2", "R2+T2", "switch(R2+T2)", "R2+T2+K40", "R0+T2+K40", "R0+T2+K40+G", "R2+T2+K40+G", "R2+T2+K40+G+RD"]:
        rs = runs(name)
        f = lambda s, b: np.mean([r[s]["overall"][b]["APCD"] for r in rs]) if rs and s in rs[0] else float("nan")
        print(f"| {name} | {f('val', 'all'):.3f} | {f('val', '30-50'):.3f} | {f('val', '50-200'):.3f} | {f('select', 'all'):.3f} | {f('select', '50-200'):.3f} |")


if __name__ == "__main__":
    main()
