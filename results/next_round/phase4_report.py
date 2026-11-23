"""Phase 4 report: the gated model on select by the rule of PLAN.md (with the camera-removal veto), val once for the
configuration it picked, and the gate values.

    python results/next_round/phase4_report.py > results/next_round/phase4/report.md
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
P1, P2, P3, P4 = (HERE / f"phase{k}" for k in (1, 2, 3, 4))
GROUPS = ("overall", "highway", "urban", "night", "rain", "clear", "adverse")
BINS = ("all", "0-30", "30-50", "50-100", "100-200", "50-200")
METRICS = ("APCD", "APBEV@0.3", "AP3D@0.5", "recall@BEV0.3")
REF = "R2+T2+K40"


def load(paths, name):
    out = []
    for p in paths:
        if p.exists():
            out += json.load(open(p)).get(name, {}).get("runs", [])
    return sorted(out, key=lambda r: r["seed"])


def arr(rs, split, g, b, metric="APCD"):
    return np.array([r[split][g][b][metric] for r in rs])


def ms(a):
    return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"


def main():
    dec = json.load(open(P4 / "decision.json"))
    print("## Candidates on select (seeds 1-3, APCD)\n")
    print("| Configuration | Params | 0-30 | 30-50 | 50-100 | 100-200 | 50-200 | Δ 50-200 | Δ 100-200 | Beats reference | "
          "No camera: 50-200 | No camera: 100-200 | Vetoed | Beats switch |")
    print("|" + "---|" * 14)
    ref_runs = load([P3 / "stageA.json"], REF)[:3]
    ref_nc = load([P4 / "nocam_ref.json"], REF)
    rows = [(REF, ref_runs, ref_nc, None)] + [(c, load([P4 / "candidates.json"], c), load([P4 / "nocam.json"], c), d)
                                             for c, d in dec["candidates"].items()]
    for name, rs, nc, d in rows:
        a = {b: arr(rs, "select", "overall", b) for b in BINS}
        n = {b: arr(nc, "select", "overall", b) for b in ("50-200", "100-200")}
        delta = "| | " if d is None else f"| {d['reference']['50-200']['margin']:+.4f} | {d['reference']['100-200']['margin']:+.4f} "
        print(f"| {name} | {rs[0].get('params', 2161)} | " + " | ".join(f"{a[b].mean():.3f}" for b in ("0-30", "30-50", "50-100", "100-200")) +
              f" | {ms(a['50-200'])} {delta}| {'-' if d is None else 'yes' if d['beats_reference'] else 'no'} | "
              f"{ms(n['50-200'])} | {ms(n['100-200'])} | {'-' if d is None else 'yes' if d['vetoed'] else 'no'} | "
              f"{'-' if d is None else 'yes' if d['beats_switch'] else 'no'} |")
    print(f"\nFinal configuration: **{dec['final']}**.\n")
    final = dec["final"]

    gates = P4 / "gates_select.json"
    for split in ("select", "val"):
        path = P4 / f"gates_{split}.json"
        if not path.exists():
            continue
        g = json.load(open(path))
        for name, res in g.items():
            print(f"### Gates, {split}, {name}: mean g_cam / g_radar for true cars (n)\n")
            print("| Condition | 0-30 | 30-50 | 50-100 | 100-200 |")
            print("|---|---|---|---|---|")
            for odd in ("highway", "urban", "night", "rain"):
                cells = []
                for b in ("0-30", "30-50", "50-100", "100-200"):
                    e = res.get(f"{odd}|{b}|True")
                    cells.append(f"{e['g_cam']:.2f} / {e['g_radar']:.2f} ({e['n']:,})" if e else "-")
                print(f"| {odd} | " + " | ".join(cells) + " |")
            print()

    if final == REF:
        print("Nothing was adopted, so val is not scored again (Phase 3 has R2+T2+K40 on val).")
        return
    r0 = final.replace("R2+T2", "R0+T2")
    sec = json.load(open(P1 / "detectors_fit/aimotive_second.json"))
    f1 = [P1 / "final_s123.json", P1 / "final_s456.json"]
    f3 = [P3 / "final_s123.json", P3 / "final_s456.json"]
    models = {"R0+T2": load(f1, "R0+T2"), "R2+T2": load(f1, "R2+T2"), "switch(R2+T2)": load([P2 / "switch.json"], "switch(R2+T2)"),
              REF: load(f3, REF), r0: load([P4 / "final_r0_s123.json", P4 / "final_s456.json"], r0),
              final: load([P4 / "final_s123.json", P4 / "final_s456.json"], final)}
    names = list(models)
    for split in ("select", "val"):
        print(f"## {split}: APCD by condition and distance (mean of 6 seeds; radar gain of the final model seed against seed)\n")
        print("| Condition | Distance | Cars | SECOND | " + " | ".join(names) +
              f" | final − SECOND | final − switch | final − {REF} | radar gain |")
        print("|" + "---|" * (8 + len(names)))
        for g in GROUPS:
            for b in BINS:
                e = sec[split].get(g, {}).get(b)
                if not isinstance(e, dict) or not e.get("n_gt"):
                    continue
                fa = arr(models[final], split, g, b)
                d = fa - arr(models[r0], split, g, b)
                print(f"| {g} | {b} | {e['n_gt']:,} | {e['APCD']:.3f} | " + " | ".join(f"{arr(models[m], split, g, b).mean():.3f}" for m in names) +
                      f" | {fa.mean() - e['APCD']:+.3f} | {fa.mean() - arr(models['switch(R2+T2)'], split, g, b).mean():+.3f} | "
                      f"{fa.mean() - arr(models[REF], split, g, b).mean():+.3f} | "
                      f"{d.mean():+.3f} ({d.min():+.3f} to {d.max():+.3f}, {int((d > 0).sum())} of {len(d)}) |")
        print(f"\n### {split}: other metrics, overall beyond 50 m\n")
        print("| Metric | SECOND | " + " | ".join(names) + " |")
        print("|" + "---|" * (2 + len(names)))
        for metric in METRICS:
            print(f"| {metric} | {sec[split]['overall']['50-200'][metric]:.3f} | " +
                  " | ".join(f"{arr(models[m], split, 'overall', '50-200', metric).mean():.3f}" for m in names) + " |")
        print()
        for m in (final, r0, REF):
            print(f"{split} {m} beyond 50 m: {ms(arr(models[m], split, 'overall', '50-200'))}; 100-200 m: {ms(arr(models[m], split, 'overall', '100-200'))}  ")
        print()


if __name__ == "__main__":
    main()
