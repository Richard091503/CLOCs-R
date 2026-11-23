"""Phase 1 report: detectors trained on fit, fusion with in-sample and cross-fitted training inputs on select,
the rule of PLAN.md, and val once for the chosen arm.

    python results/next_round/phase1_report.py > results/next_round/phase1/report.md
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
P1, P0 = HERE / "phase1", HERE / "phase0"
MODELS = ("R0+T2", "R2+T2")
GROUPS = ("overall", "highway", "urban", "night", "rain", "clear", "adverse")
BINS = ("0-30", "30-50", "50-100", "100-200", "50-200")
METRICS = ("APCD", "APBEV@0.3", "AP3D@0.5", "recall@BEV0.3")


def load(*paths):
    out = {m: [] for m in MODELS}
    for p in paths:
        if p.exists():
            d = json.load(open(p))
            for m in MODELS:
                out[m] += d.get(m, {}).get("runs", [])
    return {m: sorted(r, key=lambda x: x["seed"]) for m, r in out.items()}


def arr(runs, split, g, b, metric="APCD"):
    return np.array([r[split][g][b][metric] for r in runs if isinstance(r.get(split, {}).get(g), dict) and b in r[split][g]])


def gain(runs, split, g, b):
    k = min(len(runs["R0+T2"]), len(runs["R2+T2"]))
    d = arr(runs["R2+T2"][:k], split, g, b) - arr(runs["R0+T2"][:k], split, g, b)
    return f"{np.mean(d):+.3f} ({np.min(d):+.3f} to {np.max(d):+.3f}, {int((d > 0).sum())} of {len(d)})" if len(d) else "-"


def table(split, columns, sec):
    """columns: list of (title, runs-by-model); sec: SECOND-alone json for this split's detectors."""
    head = "| Condition | Distance | Cars | SECOND | " + " | ".join(f"{t} {m}" for t, _ in columns for m in MODELS) + \
           " | " + " | ".join(f"{t} radar gain" for t, _ in columns) + " |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for g in GROUPS:
        for b in BINS:
            e = sec[split].get(g, {}).get(b)
            if not isinstance(e, dict) or not e.get("n_gt"):
                continue
            cells = [f"{np.mean(arr(r[m], split, g, b)):.3f}" for _, r in columns for m in MODELS]
            print(f"| {g} | {b} | {e['n_gt']:,} | {e['APCD']:.3f} | " + " | ".join(cells) + " | " +
                  " | ".join(gain(r, split, g, b) for _, r in columns) + " |")
    print()


def main():
    sec = json.load(open(P1 / "detectors_fit/aimotive_second.json"))
    cas = json.load(open(P1 / "detectors_fit/aimotive_cascade.json"))
    ins, cf = load(P1 / "select_insample.json"), load(P1 / "select_crossfit.json")
    dec = json.load(open(P1 / "decision.json")) if (P1 / "decision.json").exists() else None
    final = load(P1 / "final_s123.json", P1 / "final_s456.json")

    print("## Detectors alone (trained on fit)\n")
    print("| Split | SECOND APCD > 50 m | SECOND APCD 100-200 m | Cascade AP2D@0.5 | Cascade recall@0.5, 100-200 m |")
    print("|---|---|---|---|---|")
    for s in ("select", "val"):
        o = cas[s]["overall"]
        print(f"| {s} | {sec[s]['overall']['50-200']['APCD']:.3f} | {sec[s]['overall']['100-200']['APCD']:.3f} | "
              f"{o['ap']['all']['AP2D@0.5']:.3f} | {o['recall']['100-200']['recall@0.5']:.3f} |")
    print()

    print("## Rule (select, APCD beyond 50 m, seeds 1-3)\n")
    print("| Model | In-sample | Cross-fitted | Difference | SD of cross-fitted |")
    print("|---|---|---|---|---|")
    for m in ("R2+T2", "R0+T2"):
        a, b = arr(ins[m], "select", "overall", "50-200"), arr(cf[m], "select", "overall", "50-200")
        print(f"| {m} | {a.mean():.4f} {np.round(a, 4).tolist()} | {b.mean():.4f} {np.round(b, 4).tolist()} | "
              f"{b.mean() - a.mean():+.4f} | {b.std(ddof=1):.4f} |")
    if dec:
        print(f"\nRule chose: **{dec['arm']}**.\n")

    print("## select: by condition and distance (seeds 1-3)\n")
    table("select", [("in-sample", ins), ("cross-fitted", cf)], sec)

    if all(final[m] for m in MODELS):
        print(f"## val, once: {dec['arm'] if dec else '?'} arm, {len(final['R2+T2'])} seeds\n")
        table("val", [(dec["arm"] if dec else "final", final)], sec)
        print("## Other metrics, val, overall beyond 50 m\n")
        print("| Metric | SECOND | R0+T2 | R2+T2 |")
        print("|---|---|---|---|")
        for metric in METRICS:
            print(f"| {metric} | {sec['val']['overall']['50-200'][metric]:.3f} | " +
                  " | ".join(f"{arr(final[m], 'val', 'overall', '50-200', metric).mean():.3f}" for m in MODELS) + " |")
        print("\n## val beyond 50 m: Phase 0 detectors (trained on train) against this phase's (trained on fit)\n")
        p0 = load(P0 / "fusion_s123.json", P0 / "fusion_s456.json")
        s0 = json.load(open(P0 / "detectors/aimotive_second.json"))
        print("| Model | Phase 0 | Phase 1 |")
        print("|---|---|---|")
        print(f"| SECOND | {s0['val']['overall']['50-200']['APCD']:.3f} | {sec['val']['overall']['50-200']['APCD']:.3f} |")
        for m in MODELS:
            a, b = arr(p0[m], "val", "overall", "50-200"), arr(final[m], "val", "overall", "50-200")
            print(f"| {m} | {a.mean():.4f} ± {a.std(ddof=1):.4f} | {b.mean():.4f} ± {b.std(ddof=1):.4f} |")
        print(f"| radar gain | {gain(p0, 'val', 'overall', '50-200')} | {gain(final, 'val', 'overall', '50-200')} |")


if __name__ == "__main__":
    main()
