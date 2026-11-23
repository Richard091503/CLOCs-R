"""Phase 0 report: the retrained pipeline on this machine against the RTX 4090 results, with the rule of PLAN.md.

    python results/next_round/phase0_report.py            # prints markdown, writes results/next_round/phase0/summary.json
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
NEW = ROOT / "results/next_round/phase0"
OLD = ROOT / "results/full_dataset"
MODELS = ("R0+T2", "R2+T2")
GROUPS = ("overall", "highway", "urban", "night", "rain", "clear", "adverse")
BINS = ("0-30", "30-50", "50-100", "100-200", "50-200")
METRICS = ("APCD", "APBEV@0.3", "AP3D@0.5", "recall@BEV0.3")


def runs(paths, model):
    out = []
    for p in paths:
        if p.exists():
            out += json.load(open(p)).get(model, {}).get("runs", [])
    return sorted(out, key=lambda r: r["seed"])


def get(run_or_det, split, group, b, metric="APCD"):
    return run_or_det[split][group][b][metric]


def seeds(rs, split, group, b, metric="APCD"):
    return np.array([get(r, split, group, b, metric) for r in rs])


def fmt(x):
    return f"{x:.3f}"


def main():
    new = {m: runs([NEW / "fusion_s123.json", NEW / "fusion_s456.json"], m) for m in MODELS}
    old = {m: runs([OLD / "ablation/val.json", OLD / "ablation/extra_seeds.json"], m) for m in MODELS}
    committed = {m: runs([NEW / "committed_4090_weights.json"], m) for m in MODELS}
    sec_new = json.load(open(NEW / "detectors/aimotive_second.json"))
    sec_old = json.load(open(OLD / "detectors/aimotive_second.json"))
    summary = dict(seeds={m: [r["seed"] for r in new[m]] for m in MODELS})

    print("## Rule (val, center-distance AP beyond 50 m)\n")
    print("| Model | Reference mean ± SD | New mean ± SD | Difference | 2 SE limit | Reproduces |")
    print("|---|---|---|---|---|---|")
    rule = {}
    for m in MODELS:
        a, b = seeds(old[m], "val", "overall", "50-200"), seeds(new[m], "val", "overall", "50-200")
        lim = 2 * np.sqrt(a.std(ddof=1) ** 2 / len(a) + b.std(ddof=1) ** 2 / len(b))
        ok = bool(abs(b.mean() - a.mean()) <= lim)
        rule[m] = dict(ref=a.tolist(), new=b.tolist(), diff=float(b.mean() - a.mean()), limit=float(lim), reproduces=ok)
        print(f"| {m} | {a.mean():.4f} ± {a.std(ddof=1):.4f} | {b.mean():.4f} ± {b.std(ddof=1):.4f} | {b.mean() - a.mean():+.4f} | "
              f"{lim:.4f} | {'yes' if ok else 'no'} |")
    summary["rule"] = rule
    print(f"\nPhase 0 {'passes' if all(r['reproduces'] for r in rule.values()) else 'fails'} "
          f"(seeds: {summary['seeds']}).\n")

    for split in ("val", "holdout"):
        print(f"## {split}: center-distance AP by condition and distance (4090 → this run; fusion = mean of seeds)\n")
        print("| Condition | Distance | Cars | SECOND | R0+T2 | R2+T2 | Radar gain (min to max, pairs > 0) |")
        print("|---|---|---|---|---|---|---|")
        for g in GROUPS:
            for b in BINS:
                if b not in sec_new[split].get(g, {}):
                    continue
                n = sec_new[split][g][b]["n_gt"]
                if not n:
                    continue
                s = f"{fmt(get(sec_old, split, g, b))} → {fmt(get(sec_new, split, g, b))}"
                cols = []
                for m in MODELS:
                    cols.append(f"{fmt(np.nanmean(seeds(old[m], split, g, b)))} → {fmt(np.nanmean(seeds(new[m], split, g, b)))}")
                gains = {}
                for tag, rs in (("old", old), ("new", new)):
                    k = min(len(rs["R0+T2"]), len(rs["R2+T2"]))
                    gains[tag] = seeds(rs["R2+T2"][:k], split, g, b) - seeds(rs["R0+T2"][:k], split, g, b)
                gn, go = gains["new"], gains["old"]
                gain = (f"{np.nanmean(go):+.3f} → {np.nanmean(gn):+.3f} ({np.nanmin(gn):+.3f} to {np.nanmax(gn):+.3f}, "
                        f"{int((gn > 0).sum())} of {len(gn)})")
                print(f"| {g} | {b} | {n:,} | {s} | {cols[0]} | {cols[1]} | {gain} |")
        print()

    print("## Other metrics, val, overall beyond 50 m (4090 → this run)\n")
    print("| Metric | SECOND | R0+T2 | R2+T2 |")
    print("|---|---|---|---|")
    for metric in METRICS:
        row = [f"{fmt(get(sec_old, 'val', 'overall', '50-200', metric))} → {fmt(get(sec_new, 'val', 'overall', '50-200', metric))}"]
        for m in MODELS:
            row.append(f"{fmt(seeds(old[m], 'val', 'overall', '50-200', metric).mean())} → "
                       f"{fmt(seeds(new[m], 'val', 'overall', '50-200', metric).mean())}")
        print(f"| {metric} | " + " | ".join(row) + " |")

    if all(committed[m] for m in MODELS):
        print("\n## Committed 4090 fusion networks on this run's candidates (val, APCD beyond 50 m)\n")
        print("| Model | 4090 candidates (reported) | This run's candidates |")
        print("|---|---|---|")
        for m in MODELS:
            a, c = seeds(old[m], "val", "overall", "50-200"), seeds(committed[m], "val", "overall", "50-200")
            print(f"| {m} | {a.mean():.4f} | {c.mean():.4f} |")
        summary["committed"] = {m: seeds(committed[m], "val", "overall", "50-200").tolist() for m in MODELS}
    json.dump(summary, open(NEW / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
