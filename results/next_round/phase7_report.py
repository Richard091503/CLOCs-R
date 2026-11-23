"""Phase 7 report: radar detectors on their own, the camera-free experts (L0, LRf, LRd), the selector S and the post-hoc
blend B against SECOND, LC and the incumbent, on select and val, with seed-paired differences, scene-bootstrap intervals,
gate weights, the oracle ceiling, every variant tried and the val exposure log.

    python results/next_round/phase7_report.py > results/next_round/phase7/report.md
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
P1, P3, P4, P6, P7 = (HERE / f"phase{k}" for k in (1, 3, 4, 6, 7))
BINS = ("all", "0-30", "30-50", "50-100", "100-200", "50-200")
GROUPS = ("overall", "clear", "adverse", "highway", "urban", "night", "rain")
FAR = ("50-200", "100-200")


def load(files, name):
    out = []
    for f in files:
        if f.exists():
            out += json.load(open(f)).get(name, {}).get("runs", [])
    return sorted({r["seed"]: r for r in out}.values(), key=lambda r: r["seed"])


def models():
    """label -> (description, runs per split). L has one 'run' (no seeds)."""
    sec = json.load(open(P1 / "detectors_fit/aimotive_second.json"))
    lc = [P3 / "final_r0_s123.json", P3 / "final_s456.json"]
    inc = [P4 / "final_s123.json", P4 / "final_s456.json"]
    m = {"L": ("SECOND alone", {s: [{"seed": 0, s: sec[s]}] for s in ("select", "val")}),
         "L0": ("R0+T2+NC: CLOCs without camera or radar (control)",
                {"select": load([P7 / "experts.json"], "R0+T2+NC"), "val": load([P7 / "val_experts.json"], "R0+T2+NC")}),
         "LRf": ("R2+T2+NC: radar as point features, no camera",
                 {"select": load([P7 / "experts.json"], "R2+T2+NC"), "val": load([P7 / "val_experts.json"], "R2+T2+NC")}),
         "LRd": ("R0+T2+RP: radar through D2's boxes, no camera",
                 {"select": load([P7 / "experts_rp.json"], "R0+T2+RP"), "val": load([P7 / "val_experts_rp.json"], "R0+T2+RP")}),
         "LC": ("R0+T2+K40: camera CLOCs (Phase 3)", {"select": load(lc, "R0+T2+K40"), "val": load(lc, "R0+T2+K40")}),
         "S": ("selector over L, LC, LRd", {"select": load([P7 / "selector.json"], "S(R0+T2+RP)"),
                                            "val": load([P7 / "selector_val.json"], "S(R0+T2+RP)")}),
         "B": ("fixed blend of L, LC, LRd (post hoc)", {"select": load([P7 / "blend.json"], "B(R0+T2+RP)"),
                                                         "val": load([P7 / "blend_val.json"], "B(R0+T2+RP)")}),
         "inc": ("R2+T2+K40+G: incumbent before Phase 7", {"select": load(inc, "R2+T2+K40+G"), "val": load(inc, "R2+T2+K40+G")})}
    return m


def arr(runs, split, g, b):
    return np.array([r[split][g][b]["APCD"] for r in runs if g in r[split] and b in r[split][g]])


def cell(a):
    if len(a) == 0:
        return "-"
    return f"{a.mean():.4f}" if len(a) == 1 else f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"


def paired(a, b):
    if len(b) == 1:
        d = a - b[0]
    elif len(a) == 1:
        d = a[0] - b
    else:
        d = a - b
    return f"{d.mean():+.4f} ({int((d > 0).sum())}/{len(d)})"


def detectors():
    sec = json.load(open(P1 / "detectors_fit/aimotive_second.json"))["select"]["overall"]
    dets = {"D1 (Phase 6 PointPillars)": json.load(open(P6 / "radar_detector.json"))["select"]["overall"],
            "D2 (motion-corrected, clutter probability)": json.load(open(P7 / "radar_detector_d2.json"))["select"]["overall"],
            "D3 (RadarDistill, authors' code)": json.load(open(P7 / "radar_detector_d3.json"))["select"]["overall"]}
    print("## 1. Radar detectors on their own (`select`, APCD)\n")
    print("| Detector | " + " | ".join(BINS) + " |\n|" + "---|" * (1 + len(BINS)))
    for k, d in dets.items():
        print(f"| {k} | " + " | ".join(f"{d[b]['APCD']:.3f}" if b in d else "-" for b in BINS) + " |")
    print("| SECOND, for reference | " + " | ".join(f"{sec[b]['APCD']:.3f}" for b in BINS) + " |")
    print("\nBy the pre-registered rule, the radar detector with the highest 50–200 m APCD on `select` feeds RP: **D2**.")
    print("The radar detectors were not scored alone on val.\n")


def table(m, split):
    print(f"### {split}\n")
    print("| Model | " + " | ".join(BINS) + " |\n|" + "---|" * (1 + len(BINS)))
    for lab, (_, runs) in m.items():
        print(f"| {lab} | " + " | ".join(cell(arr(runs[split], split, "overall", b)) for b in BINS) + " |")
    print()


def conditions(m, split):
    print(f"### {split}, by condition (50–200 m / 100–200 m)\n")
    print("| Model | " + " | ".join(GROUPS) + " |\n|" + "---|" * (1 + len(GROUPS)))
    for lab, (_, runs) in m.items():
        cells = []
        for g in GROUPS:
            a, b = arr(runs[split], split, g, "50-200"), arr(runs[split], split, g, "100-200")
            cells.append(f"{a.mean():.3f} / {b.mean():.3f}" if len(a) and len(b) else "-")
        print(f"| {lab} | " + " | ".join(cells) + " |")
    print()


COMPARISONS = [("LRd", "L0"), ("LRf", "L0"), ("LRd", "LRf"), ("LRd", "L"), ("LRf", "L"), ("S", "LC"), ("S", "L"), ("S", "inc"),
               ("S", "B"), ("B", "LC"), ("B", "L"), ("B", "inc"), ("LC", "L")]


def seed_paired(m):
    print("## 3. Seed-paired differences (mean, and seeds where the first is higher)\n")
    print("L has no seeds, so differences against L are each seed against the one SECOND value.\n")
    print("| Comparison | " + " | ".join(f"{s} {b}" for s in ("select", "val") for b in ("0-30", "30-50") + FAR) + " |")
    print("|" + "---|" * 9)
    for a, b in COMPARISONS:
        cells = []
        for s in ("select", "val"):
            for bn in ("0-30", "30-50") + FAR:
                cells.append(paired(arr(m[a][1][s], s, "overall", bn), arr(m[b][1][s], s, "overall", bn)))
        print(f"| {a} − {b} | " + " | ".join(cells) + " |")
    print()


def rule():
    d = json.load(open(P7 / "decision.json"))
    print("## 4. The pre-registered rule (`select`)\n")
    print("A model replaces the incumbent only if its mean beats the incumbent's by more than its own seed SD at both "
          "50–200 m and 100–200 m; among those, the highest 50–200 m mean wins.\n")
    print("| Candidate | 50–200 mean ± SD | incumbent | 100–200 mean ± SD | incumbent | beats |\n|---|---|---|---|---|---|")
    for k, v in d["candidates"].items():
        a, b = v["50-200"], v["100-200"]
        print(f"| {k} | {a['mean']:.4f} ± {a['sd']:.4f} | {a['incumbent']:.4f} | {b['mean']:.4f} ± {b['sd']:.4f} | "
              f"{b['incumbent']:.4f} | {'yes' if v['beats'] else 'no'} |")
    print(f"\nCurrent model by the rule: **{d['current']}**. B was defined after these results and is not a candidate.\n")


def bootstrap():
    print("## 5. Scene-bootstrap 95% intervals (1,000 scene resamples, means over 6 seeds)\n")
    for split in ("select", "val"):
        bs = json.load(open(P7 / f"bootstrap_{split}.json"))
        print(f"### {split} ({bs['scenes']} scenes, {bs['frames']} frames)\n")
        bins = ("50-100", "100-200", "50-200", "all")
        print("| Comparison | " + " | ".join(bins) + " |\n|" + "---|" * (1 + len(bins)))
        for k, v in bs["comparisons"].items():
            print(f"| {k} | " + " | ".join(f"{v[b]['diff']:+.4f} [{v[b]['lo']:+.4f}, {v[b]['hi']:+.4f}]" for b in bins) + " |")
        print()


def gates(m):
    print("## 6. Expert weights\n")
    print("Mean weights of candidates with SECOND score ≥ 0.05, by condition and distance (L, LC, LRd).\n")
    for split in ("select", "val"):
        runs = m["S"][1][split]
        keys = sorted({k for r in runs for k in r[split + "_weights"]}, key=lambda k: (k.split("|")[0], int(k.split("|")[1].split("-")[0])))
        print(f"### S on {split}\n")
        print("| Condition, bin | L | LC | LRd | candidates |\n|---|---|---|---|---|")
        for k in keys:
            w = np.mean([[r[split + "_weights"][k][e] for e in ("L", "LC", "LR")] for r in runs if k in r[split + "_weights"]], 0)
            n = int(np.mean([r[split + "_weights"][k]["n"] for r in runs if k in r[split + "_weights"]]))
            print(f"| {k} | {w[0]:.3f} | {w[1]:.3f} | {w[2]:.3f} | {n} |")
        print()
    c = json.load(open(P7 / "ceiling.json"))
    bw = np.array([r["B_weights"] for r in c["runs"]])
    print("### B (the same for every candidate)\n")
    print("| Seed | L | LC | LRd |\n|---|---|---|---|")
    for r in c["runs"]:
        print(f"| {r['seed']} | " + " | ".join(f"{x:.3f}" for x in r["B_weights"]) + " |")
    print(f"| mean | " + " | ".join(f"{x:.3f}" for x in bw.mean(0)) + " |\n")
    print("![S's weights by distance](../figures/phase7_weights.png)\n")


def ceiling():
    c = json.load(open(P7 / "ceiling.json"))
    names = ("0-30", "30-50", "50-100", "100-200", "50-200")
    print("## 7. Oracle ceiling of situation-dependent blends (`select`; upper bounds tuned on `select`)\n")
    print(f"{c['grid_points']} weightings on the simplex (step {c['grid_step']}), 6 seeds. Each oracle minus B:\n")
    print("| Upper bound | " + " | ".join(names) + " |\n|" + "---|" * (1 + len(names)))
    for k, lab in (("global_oracle", "best single w per bin"), ("distance_oracle", "best w per distance bin"),
                   ("distance_condition_oracle", "best w per distance bin and condition (annotation labels)")):
        print(f"| {lab} | " + " | ".join(f"{c['mean'][k][n] - c['mean']['B'][n]:+.4f}" for n in names) + " |")
    print("| distance oracle − global oracle | " + " | ".join(f"{c['distance_minus_global'][n]:+.4f}" for n in names) + " |")
    print(f"\nRule: further selector variants only if the distance oracle beats B by ≥ 0.005 at 100–200 m. It is "
          f"{c['distance_oracle_minus_B']:+.4f}, so **{'they run' if c['proceed'] else 'selector work stops'}**.\n")


def figure(m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"L": "#2a78d6", "LC": "#eb6834", "LR": "#1baf7a"}
    names = {"L": "L (SECOND alone)", "LC": "LC (camera CLOCs)", "LR": "LRd (radar-detector CLOCs)"}
    bins = ("0-30", "30-50", "50-100", "100-200")
    c = json.load(open(P7 / "ceiling.json"))
    bw = np.array([r["B_weights"] for r in c["runs"]]).mean(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, split in zip(axes, ("select", "val")):
        runs = m["S"][1][split]
        for i, e in enumerate(("L", "LC", "LR")):
            ys = []
            for b in bins:
                ws = [r[split + "_weights"][k][e] * r[split + "_weights"][k]["n"] for r in runs for k in r[split + "_weights"] if k.endswith("|" + b)]
                ns = [r[split + "_weights"][k]["n"] for r in runs for k in r[split + "_weights"] if k.endswith("|" + b)]
                ys.append(sum(ws) / sum(ns))
            ax.plot(range(len(bins)), ys, color=colors[e], lw=2, marker="o", ms=6, label=names[e] + ", S")
            ax.axhline(bw[i], color=colors[e], lw=1.2, ls="--", label=names[e] + ", B (fixed)")
        ax.set_xticks(range(len(bins)), [b + " m" for b in bins])
        ax.set_title(f"S's expert weights by distance, {split}\n(6 seeds; candidates with SECOND score ≥ 0.05)", fontsize=9.5)
        ax.grid(axis="y", color="#e5e5e2", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("mean weight")
    axes[1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    out = HERE / "figures/phase7_weights.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")


def variants():
    print("## 8. Every variant tried in Phase 7\n")
    print("""- **Radar input** for the radar detector, chosen on `select` by the rule: D2's inputs (motion-corrected sweeps and the
  clutter probability) over D1's.
- **Radar detectors:** D1, D2 and D3 (RadarDistill: teacher 10 epochs, student 20, halved by the pre-registered
  24-hour rule). D2 chosen.
- **Radar experts:** LRf (R2+T2+NC) and LRd (R0+T2+RP), each 6 seeds, against the control L0 (R0+T2+NC). LRd chosen as
  S's radar expert by the rule.
- **Selector S** over L, LC, LRd: 6 seeds, out-of-fold stacking on `fit`'s 3 scene folds.
- **Post hoc, labelled as such:**
  - the fixed blend B, 6 seeds, defined after the `select` results;
  - the oracle ceiling (diagnostic, upper bounds on `select` only, never on val).
- **Considered and not run:**
  - after the ceiling, by its rule: S plus radar-detector evidence, and a selector over the three detectors' own scores;
  - by the user's decision: a per-distance calibration of the fused score (DECISIONS.md, 2026-11-21, item 9).
""")


def exposure():
    print("## 9. Val exposure\n")
    print("""- Val was looked at in every earlier phase, so it is a development set (REEVALUATION.md, section 5).
- **In Phase 7, val was scored once,** on 2026-11-21 (`phase7_val.sh`, 12:50–18:58 UTC), after every model and B
  were frozen. It was scored for L0, LRf, LRd, S and B from saved weights, plus the scene bootstrap. LC, the incumbent
  and SECOND are earlier models whose val results were already known.
- No setting was changed after val was seen. The ceiling check ran after val, on `select` only, as a diagnostic.
""")


def main():
    m = models()
    figure(m)
    print("# Phase 7: radar through a detector, and a situational selector\n")
    print("Pre-registered in PLAN.md (Phase 7 and its addenda); decisions and interpretations in DECISIONS.md "
          "(2026-11-21). All numbers are APCD, means over 6 seeds (± seed SD), except SECOND alone.\n")
    print("| Label | Model |\n|---|---|")
    for lab, (desc, _) in m.items():
        print(f"| {lab} | {desc} |")
    print()
    detectors()
    print("## 2. Models on `select` and val\n")
    for s in ("select", "val"):
        table(m, s)
    for s in ("select", "val"):
        conditions(m, s)
    seed_paired(m)
    rule()
    bootstrap()
    gates(m)
    ceiling()
    variants()
    exposure()
    print("## Reproduce\n")
    print("""```bash
bash results/next_round/phase7_run.sh      # radar detectors, experts, selector, the rule (select only)
bash results/next_round/phase7_val.sh      # B on select, val once for every model, scene bootstrap
python -W ignore -m clocs_r.selector_ceiling --radar_dets runs/next_round/phase7/d2_dets --out results/next_round/phase7/ceiling.json
python results/next_round/phase7_report.py > results/next_round/phase7/report.md
```""")


if __name__ == "__main__":
    main()
