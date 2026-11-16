"""Tables and the figure for results/full_dataset/README.md, from the outputs of clocs_r/run_full.sh and per_scene.py.

    python results/full_dataset/make_report.py [--dir results/full_dataset]
Writes tables.md and figures/far_by_condition.png, comparing the model chosen on holdout (selected.json) with its radar-free twin.
Every fusion number is the mean over seeds with the range in brackets.

To report another configuration, with extra seeds trained later (tune_fusion --seeds) merged in:
    python results/full_dataset/make_report.py --model R3+T2 --twin R0+T2 --extra rerun_r3.json,rerun_r2.json --suffix _r3
which writes tables_r3.md and figures/far_by_condition_r3.png, and reads per_scene_val_r3.json if present.
"""
import argparse
import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINS = ("all", "0-30", "30-50", "50-100", "100-200", "50-200")
BIN_NAMES = {"all": "All", "0-30": "0-30 m", "30-50": "30-50 m", "50-100": "50-100 m", "100-200": "100-200 m", "50-200": "**Beyond 50 m**"}
GROUPS = ("overall", "clear", "highway", "urban", "adverse", "night", "rain", "no_tunnels")
GROUP_NAMES = {"overall": "All conditions", "clear": "Clear (highway + urban)", "highway": "Highway", "urban": "Urban",
               "adverse": "Adverse (night + rain)", "night": "Night", "rain": "Rain", "no_tunnels": "Without tunnel scenes"}
# Reference palette slots 1-3 (dataviz skill), chart chrome and ink, light mode
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def seed_values(entry, split, group, bin_, metric):
    return [r[split][group][bin_][metric] for r in entry["runs"] if group in r[split]]


def fmt(values, digits=3, rng=True):
    if not values:
        return "n/a"
    m = f"{np.mean(values):.{digits}f}"
    return f"{m} ({min(values):.{digits}f}-{max(values):.{digits}f})" if rng and len(values) > 1 else m


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(pathlib.Path(__file__).parent))
    ap.add_argument("--model", default=None, help="CLOCs-R configuration to report (default: the one selected on holdout)")
    ap.add_argument("--twin", default=None, help="its radar-free twin (default: from selected.json)")
    ap.add_argument("--extra", default="", help="comma-separated tune_fusion outputs whose seeds are added to final.json's")
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    ap.add_argument("--final", default="final.json", help="tune_fusion output with holdout and val scores of every column")
    ap.add_argument("--far1", default="final_far_weight_1.json", help="R0 trained without range weighting ('' to leave out)")
    ap.add_argument("--selected", default="selected.json", help="selection to show first ('' to leave out; then --model and --twin are needed)")
    ap.add_argument("--columns", default="", help="comma-separated fusion columns (default: every configuration in --final)")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    sel = json.load(open(d / args.selected)) if args.selected else dict(winner=args.model, twin=args.twin, holdout={})
    final = json.load(open(d / args.final))
    if args.far1:
        final.update({"R0 (far weight 1)": v for v in json.load(open(d / args.far1)).values()})
    second = json.load(open(d / "detectors" / "aimotive_second.json"))
    cascade = json.load(open(d / "detectors" / "aimotive_cascade.json"))
    for extra in filter(None, args.extra.split(",")):
        for name, entry in json.load(open(d / extra)).items():
            have = {r["seed"] for r in final[name]["runs"]}
            assert entry["settings"] == final[name]["settings"] and entry["far_weight"] == final[name]["far_weight"], name
            final[name]["runs"] += [r for r in entry["runs"] if r["seed"] not in have]
    win, twin = args.model or sel["winner"], args.twin or sel["twin"]
    n_seeds = len(final[win]["runs"])
    chosen = "selected" if win == sel["winner"] else "reported"
    more = [f"{k} ({len(v['runs'])} seeds)" for k, v in final.items() if len(v["runs"]) != 3]
    seed_note = "Fusion columns: mean of 3 seeds" + (", except " + ", ".join(more) if more else "") + ".\n"
    md = []

    if sel["holdout"]:
        md.append("## Selection on holdout\n\nCenter-distance AP beyond 50 m, mean of 3 seeds. The CLOCs-R candidates competed; "
                  "the radar-free rows are for reference.\n")
        rows = [(f"**{n}**" if n == sel["winner"] else n, f"{v['mean']:.4f}", ", ".join(f"{x:.4f}" for x in v["seeds"]),
                 "selected" if n == sel["winner"] else ("candidate" if n in ("R2", "R3", "R2+T2", "R3+T2") else "no radar"))
                for n, v in sorted(sel["holdout"].items(), key=lambda kv: -kv[1]["mean"])]
        sec = second["holdout"]["overall"]["50-200"]["APCD"]
        rows.append(("SECOND alone", f"{sec:.4f}", "", "detector"))
        md.append(table(("Configuration", "Mean", "Seeds", "Role"), rows))

    names = args.columns.split(",") if args.columns else [k for k in final if k != "R0 (far weight 1)"]
    models = [("SECOND", None)] + ([("CLOCs R0 (far weight 1)", "R0 (far weight 1)")] if args.far1 else []) + \
             [(("CLOCs " + n) if n.startswith("R0") else n, n) for n in names]

    def value(split, key, group, bin_, metric, rng=True):
        if key is None:
            v = second[split].get(group, {}).get(bin_, {}).get(metric)
            return "n/a" if v is None else f"{v:.3f}"
        return fmt(seed_values(final[key], split, group, bin_, metric), rng=rng)

    for metric, title in (("APCD", "Center-distance AP"), ("APBEV@0.3", "BEV AP@0.3"), ("AP3D@0.5", "AP3D@0.5"),
                          ("recall@BEV0.3", "Recall at BEV IoU 0.3")):
        md.append(f"\n## {title}, val, all conditions\n\nCLOCs-R ({chosen}): **{win}**; its radar-free twin: **{twin}**. "
                  + seed_note)
        header = ["Distance", "n GT"] + [f"**{n}**" if k in (win, twin) else n for n, k in models]
        rows = [[BIN_NAMES[b], second["val"]["overall"][b]["n_gt"]] + [value("val", k, "overall", b, metric, rng=False) for _, k in models]
                for b in BINS]
        md.append(table(header, rows))

    md.append(f"\n## By driving condition, val: center-distance AP beyond 50 m\n\nRadar gain is {win} minus {twin}, by seed "
              f"(seed i against seed i), mean of {n_seeds} pairs with the range.\n")
    header = ["Condition", "Cars beyond 50 m", "SECOND", "CLOCs R0", f"{twin}", f"**{win}**", "Radar gain", "vs SECOND"]
    rows = []
    for g in GROUPS:
        if g not in second["val"]:
            continue
        w = seed_values(final[win], "val", g, "50-200", "APCD")
        t = seed_values(final[twin], "val", g, "50-200", "APCD")
        s = second["val"][g]["50-200"]["APCD"]
        gain = [a - b for a, b in zip(w, t)]
        rows.append([GROUP_NAMES[g], second["val"][g]["50-200"]["n_gt"], f"{s:.3f}", value("val", "R0", g, "50-200", "APCD", False),
                     fmt(t, rng=False), fmt(w, rng=False), fmt(gain, digits=3), f"{np.mean(w) - s:+.3f}"])
    md.append(table(header, rows))

    for metric, title in (("APCD", "center-distance AP"), ("APBEV@0.3", "BEV AP@0.3")):
        md.append(f"\n## Radar gain by condition and distance, val ({title})\n\n{win} minus {twin}, mean of the {n_seeds} seed pairs "
                  "with the range. Cars per cell in brackets after the condition.\n")
        header = ["Condition"] + [BIN_NAMES[b].strip("*") for b in BINS]
        rows = []
        for g in GROUPS:
            if g not in second["val"]:
                continue
            cells = []
            for b in BINS:
                w = seed_values(final[win], "val", g, b, metric)
                t = seed_values(final[twin], "val", g, b, metric)
                n = second["val"][g][b]["n_gt"]
                cells.append(f"{fmt([a - c for a, c in zip(w, t)])} [{n}]" if n else "no cars")
            rows.append([GROUP_NAMES[g]] + cells)
        md.append(table(header, rows))

    md.append("\n## Holdout, by condition: center-distance AP beyond 50 m\n\n" + ("Holdout chose the model, so it flatters it.\n" if chosen == "selected" else
              f"Holdout was used to choose among the CLOCs-R candidates ({sel['winner']} won), so it is not a clean test set.\n"))
    rows = []
    for g in GROUPS:
        if g not in second["holdout"]:
            continue
        w = seed_values(final[win], "holdout", g, "50-200", "APCD")
        t = seed_values(final[twin], "holdout", g, "50-200", "APCD")
        rows.append([GROUP_NAMES[g], second["holdout"][g]["50-200"]["n_gt"], f"{second['holdout'][g]['50-200']['APCD']:.3f}",
                     fmt(t, rng=False), fmt(w, rng=False), fmt([a - b for a, b in zip(w, t)])])
    md.append(table(["Condition", "Cars beyond 50 m", "SECOND", twin, win, "Radar gain"], rows))

    md.append("\n## Cascade R-CNN alone, val\n\nAP by box height in pixels; candidate recall by the distance of the 3D label.\n")
    rows = []
    for g in GROUPS:
        c = cascade["val"].get(g)
        if not c:
            continue
        rows.append([GROUP_NAMES[g], f"{c['ap']['all']['AP2D@0.5']:.3f}", f"{c['ap']['h<25']['AP2D@0.5']:.3f}",
                     f"{c['recall']['50-100']['recall@0.5']:.3f}", f"{c['recall']['100-200']['recall@0.5']:.3f}"])
    md.append(table(["Condition", "AP2D@0.5, all", "AP2D@0.5, under 25 px", "Recall@0.5, 50-100 m", "Recall@0.5, 100-200 m"], rows))

    scene_path = d / f"per_scene_val{args.suffix}.json"
    if scene_path.exists():
        ps = json.load(open(scene_path))["scenes"]
        md.append(f"\n## Per scene, val: center-distance AP beyond 50 m\n\nScenes with at least 50 cars beyond 50 m. "
                  f"Gain is {win} minus {twin}, mean of the {n_seeds} seed pairs.\n")
        rows = []
        for seq, info in sorted(ps["SECOND"].items(), key=lambda kv: (kv[1]["odd"], kv[0])):
            n = info["metrics"]["50-200"]["n_gt"]
            if n < 50:
                continue
            w = [ps[win][s][seq]["metrics"]["50-200"]["APCD"] for s in ps[win]]
            t = [ps[twin][s][seq]["metrics"]["50-200"]["APCD"] for s in ps[twin]]
            rows.append([info["odd"], seq, n, f"{info['metrics']['50-200']['APCD']:.3f}", fmt(t, rng=False), fmt(w, rng=False),
                         f"{np.mean(w) - np.mean(t):+.3f}"])
        md.append(table(["Condition", "Scene", "Cars beyond 50 m", "SECOND", twin, win, "Radar gain"], rows))
        gains = {}
        for r in rows:
            gains.setdefault(r[0], []).append(float(r[-1]))
        md.append("\nScenes where radar helps, per condition: " +
                  ", ".join(f"{o} {sum(g > 0 for g in v)} of {len(v)}" for o, v in sorted(gains.items())) + ".")

    (d / f"tables{args.suffix}.md").write_text("\n".join(md) + "\n")
    print("wrote", d / f"tables{args.suffix}.md")
    figure(d, second, final, win, twin, args.suffix)


def figure(d, second, final, win, twin, suffix=""):
    groups = [g for g in ("clear", "highway", "urban", "adverse", "night", "rain") if g in second["val"]]
    series = [("SECOND alone", [second["val"][g]["50-200"]["APCD"] for g in groups], None),
              (f"CLOCs without radar ({twin})", *zip(*[(np.mean(v), (min(v), max(v))) for v in
                                                       (seed_values(final[twin], "val", g, "50-200", "APCD") for g in groups)])),
              (f"CLOCs-R ({win})", *zip(*[(np.mean(v), (min(v), max(v))) for v in
                                          (seed_values(final[win], "val", g, "50-200", "APCD") for g in groups)]))]
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(groups))
    width = 0.26
    for i, (name, vals, ranges) in enumerate(series):
        pos = x + (i - 1) * (width + 0.02)
        ax.bar(pos, vals, width, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if ranges is not None:  # seed range
            lo, hi = zip(*ranges)
            ax.vlines(pos, lo, hi, color=INK, linewidth=1.2, zorder=4)
    ax.set_xticks(x, [GROUP_NAMES[g].split(" (")[0] for g in groups], color=INK_2)
    ax.set_ylabel("Center-distance AP beyond 50 m", color=INK_2)
    ax.set_ylim(0, min(1, max(max(s[1]) for s in series) + 0.05))  # bars start at zero
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title(f"Val, cars beyond 50 m, by driving condition (bars: {len(final[win]['runs'])}-seed mean, whiskers: seed range)", color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right", ncol=3, bbox_to_anchor=(1, -0.1))
    fig.tight_layout()
    (d / "figures").mkdir(exist_ok=True)
    fig.savefig(d / "figures" / f"far_by_condition{suffix}.png", dpi=130)
    print("wrote", d / "figures" / f"far_by_condition{suffix}.png")


if __name__ == "__main__":
    main()
