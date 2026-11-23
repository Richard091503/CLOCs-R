"""Phase 6 report: the radar detector on its own, R2+T2+K40+G+RD by the rule, its 6-seed final on select and val against
every comparison, and the extras (SECOND ∪ radar, recall gained, false alarms added).

    python results/next_round/phase6_report.py > results/next_round/phase6/report.md
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
P1, P2, P3, P4, P6 = (HERE / f"phase{k}" for k in (1, 2, 3, 4, 6))
GROUPS = ("overall", "highway", "urban", "night", "rain", "clear", "adverse")
ODDS = ("overall", "highway", "urban", "night", "rain")
BINS = ("all", "0-30", "30-50", "50-100", "100-200", "50-200")
METRICS = ("APCD", "APBEV@0.3", "AP3D@0.5", "recall@BEV0.3")
REF, RD, R0 = "R2+T2+K40+G", "R2+T2+K40+G+RD", "R0+T2+K40+G"


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


def paired(a, b):
    d = a - b
    return f"{d.mean():+.3f} ({d.min():+.3f} to {d.max():+.3f}, {int((d > 0).sum())} of {len(d)})"


def main():
    sec = json.load(open(P1 / "detectors_fit/aimotive_second.json"))
    det = json.load(open(P6 / "radar_detector.json"))
    print("## The radar detector on its own (APCD; SECOND alone in brackets)\n")
    for split in ("select", "val"):
        print(f"### {split}\n")
        print("| Condition | " + " | ".join(BINS) + " |")
        print("|" + "---|" * (1 + len(BINS)))
        for g in GROUPS:
            if g not in det[split]:
                continue
            cells = []
            for b in BINS:
                e = det[split][g].get(b)
                s = sec[split].get(g, {}).get(b)
                cells.append(f"{e['APCD']:.3f} ({s['APCD']:.3f})" if isinstance(e, dict) and e.get("n_gt") else "-")
            print(f"| {g} | " + " | ".join(cells) + " |")
        print("\n| Metric, beyond 50 m | Radar detector | SECOND |\n|---|---|---|")
        for m in METRICS:
            print(f"| {m} | {det[split]['overall']['50-200'][m]:.3f} | {sec[split]['overall']['50-200'][m]:.3f} |")
        print()

    dec = json.load(open(P6 / "decision.json"))
    d = dec["candidates"][RD]
    print("## The rule on select (seeds 1-3, APCD)\n")
    print("| | 50-200 | 100-200 |\n|---|---|---|")
    print(f"| {REF} (reference) | {d['reference']['50-200']['old']:.4f} | {d['reference']['100-200']['old']:.4f} |")
    print(f"| {RD} | {d['reference']['50-200']['mean']:.4f} ± {d['reference']['50-200']['sd']:.4f} | "
          f"{d['reference']['100-200']['mean']:.4f} ± {d['reference']['100-200']['sd']:.4f} |")
    print(f"| margin | {d['reference']['50-200']['margin']:+.4f} | {d['reference']['100-200']['margin']:+.4f} |")
    print(f"| no camera: reference | {d['nocam']['50-200']['old']:.4f} | {d['nocam']['100-200']['old']:.4f} |")
    print(f"| no camera: {RD} | {d['nocam']['50-200']['mean']:.4f} | {d['nocam']['100-200']['mean']:.4f} |")
    print(f"| margin over switch(R2+T2) | {d['switch']['50-200']['margin']:+.4f} | {d['switch']['100-200']['margin']:+.4f} |")
    print(f"\nBeats the reference: {'yes' if d['beats_reference'] else 'no'}; vetoed: {'yes' if d['vetoed'] else 'no'}; "
          f"beats the switch: {'yes' if d['beats_switch'] else 'no'}. Current model by the rule: **{dec['current']}**.\n")

    f1 = [P1 / "final_s123.json", P1 / "final_s456.json"]
    models = {"R0+T2": load(f1, "R0+T2"), "R2+T2": load(f1, "R2+T2"), "switch(R2+T2)": load([P2 / "switch.json"], "switch(R2+T2)"),
              "R2+T2+K40": load([P3 / "final_s123.json", P3 / "final_s456.json"], "R2+T2+K40"),
              R0: load([P4 / "final_r0_s123.json", P4 / "final_s456.json"], R0),
              REF: load([P4 / "final_s123.json", P4 / "final_s456.json"], REF),
              RD: load([P6 / "final_s123.json", P6 / "final_s456.json"], RD)}
    names = list(models)
    for split in ("select", "val"):
        print(f"## {split}: APCD by condition and distance (mean of 6 seeds; paired differences seed against seed)\n")
        print("| Condition | Distance | Cars | SECOND | " + " | ".join(names) +
              f" | RD − SECOND | RD − switch | RD − {REF} (detector) | RD − {R0} (radar gain) |")
        print("|" + "---|" * (8 + len(names)))
        for g in GROUPS:
            for b in BINS:
                e = sec[split].get(g, {}).get(b)
                if not isinstance(e, dict) or not e.get("n_gt"):
                    continue
                fa = arr(models[RD], split, g, b)
                print(f"| {g} | {b} | {e['n_gt']:,} | {e['APCD']:.3f} | " + " | ".join(f"{arr(models[m], split, g, b).mean():.3f}" for m in names) +
                      f" | {fa.mean() - e['APCD']:+.3f} | {fa.mean() - arr(models['switch(R2+T2)'], split, g, b).mean():+.3f} | "
                      f"{paired(fa, arr(models[REF], split, g, b))} | {paired(fa, arr(models[R0], split, g, b))} |")
        print(f"\n### {split}: other metrics, overall beyond 50 m\n")
        print("| Metric | SECOND | " + " | ".join(names) + " |")
        print("|" + "---|" * (2 + len(names)))
        for metric in METRICS:
            print(f"| {metric} | {sec[split]['overall']['50-200'][metric]:.3f} | " +
                  " | ".join(f"{arr(models[m], split, 'overall', '50-200', metric).mean():.3f}" for m in names) + " |")
        print()
        for m in (RD, REF, R0):
            print(f"{split} {m} beyond 50 m: {ms(arr(models[m], split, 'overall', '50-200'))}; 100-200 m: {ms(arr(models[m], split, 'overall', '100-200'))}  ")
        print()

    ex = json.load(open(P6 / "extra.json"))
    print("## Extras: SECOND ∪ radar, recall gained, false alarms added (mean over seeds)\n")
    print("\"none\" drops the radar-sourced candidates; \"union\" keeps them. Recall: scored cars with a kept box of score ≥ 0.1 "
          "within 2 m. False alarms: kept radar-sourced boxes with score ≥ 0.1 more than 2 m from every scored car.\n")
    rows = [(m, v) for m in ("SECOND", R0, REF, RD) if m in ex for v in ex[m]["runs"][0]["select"] if v != "recall_fa"]
    for split in ("select", "val"):
        print(f"### {split}\n")
        print("| Model | Variant | APCD 50-200 | APCD 100-200 | Recall 50-100 | Recall 100-200 | Recall 50-200 | "
              "Radar-sourced kept | of which true | False alarms per frame |")
        print("|" + "---|" * 10)
        for m, v in rows:
            rs = ex[m]["runs"]
            a = lambda b: np.mean([r[split][v]["overall"][b]["APCD"] for r in rs])
            rf = lambda k: np.mean([r[split][v]["recall_fa"]["overall"][k] for r in rs], axis=0)
            rec = lambda b: np.mean([r[split][v]["recall_fa"]["overall"]["recall"][b] for r in rs])
            print(f"| {m} | {v} | {a('50-200'):.4f} | {a('100-200'):.4f} | {rec('50-100'):.3f} | {rec('100-200'):.3f} | "
                  f"{rec('50-200'):.3f} | {rf('radar_kept'):,.0f} | {rf('radar_true'):,.0f} | {rf('false_alarms_per_frame'):.3f} |")
        print(f"\n#### {split}: recall at 50-200 m by condition\n")
        print("| Model | Variant | " + " | ".join(ODDS) + " |")
        print("|" + "---|" * (2 + len(ODDS)))
        for m, v in rows:
            rs = ex[m]["runs"]
            print(f"| {m} | {v} | " + " | ".join(
                f"{np.mean([r[split][v]['recall_fa'][g]['recall']['50-200'] for r in rs]):.3f}" if g in rs[0][split][v]["recall_fa"] else "-"
                for g in ODDS) + " |")
        print()
    # Reproduction: the Phase 4 models scored on the augmented frames without the radar-sourced candidates
    print("### Reproduction: Phase 4 models with the radar-sourced candidates dropped, against their stored results\n")
    print("| Model | Split | largest |Δ| APCD 50-200 |\n|---|---|---|")
    for m in (R0, REF):
        if m not in ex:
            continue
        for split in ("select", "val"):
            new = {r["seed"]: r[split]["none"]["overall"]["50-200"]["APCD"] for r in ex[m]["runs"]}
            old = {r["seed"]: r[split]["overall"]["50-200"]["APCD"] for r in models[m]}
            print(f"| {m} | {split} | {max(abs(new[s] - old[s]) for s in new):.1e} |")
    if "SECOND" in ex:
        for split in ("select", "val"):
            print(f"| SECOND | {split} | {abs(ex['SECOND']['runs'][0][split]['none']['overall']['50-200']['APCD'] - sec[split]['overall']['50-200']['APCD']):.1e} |")


if __name__ == "__main__":
    main()
