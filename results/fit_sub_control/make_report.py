"""results.json and the tables of REPORT.md, from the result files of results/fit_sub_control (every number is read,
none is typed in). The verdicts follow PREDICTIONS.md's rules.

    python results/fit_sub_control/make_report.py      # writes results.json and report_tables.md
"""
import json
import pathlib
import sys

import numpy as np

O = pathlib.Path("results/fit_sub_control")
S = pathlib.Path("runs/fit_sub_control")
SEEDS = ["1", "2", "3", "4", "5", "6"]
DIST = ["all", "0-30", "30-50", "50-100", "100-200", "50-200"]
CONDS = ["highway", "urban", "night", "rain"]
NETS = ["L0", "LRd", "LC", "CLOCs-R"]
BINS = ("50-200", "50-100", "100-200")


def label(m):
    return "SECOND" if m == "L" else m


def f3(x, sign=False):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:+.3f}" if sign else f"{x:.3f}")


def seed_values(val, m, group, b):
    per = val["results"][m]
    return np.array([per[s][group][b]["APCD"] for s in (SEEDS if m != "L" else ["0"]) if group in per[s]])


def mean_sd(v):
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None, seeds=v.tolist())


def interval_verdict(diff, lo, hi, min_diff=None):
    """Supported: the interval above zero (and diff >= min_diff); contradicted: the interval below zero."""
    if hi < 0:
        return "contradicted"
    if lo > 0 and (min_diff is None or diff >= min_diff):
        return "supported"
    return "not supported"


def main():
    val = json.load(open(O / "val.json"))
    subset = json.load(open(O / "subset.json"))
    boot = val["bootstrap"]

    def comp(a, b):
        k = f"{a} - {b}"
        return {bb: dict(diff=boot["scene"][k][bb]["diff"], scene=[boot["scene"][k][bb]["lo"], boot["scene"][k][bb]["hi"]],
                         recording=[boot["recording"][k][bb]["lo"], boot["recording"][k][bb]["hi"]]) for bb in BINS}

    order = ["L"] + [x for m in NETS for x in (f"{m}_sub", f"{m}_sel", m)]
    per_model = {m: dict(distance={b: mean_sd(seed_values(val, m, "overall", b)) for b in DIST},
                         condition_50_200={c: mean_sd(seed_values(val, m, c, "50-200")) for c in CONDS}) for m in order}

    wc = val["winning_cells"]
    cells = wc["order"]["cells"]
    shares = {m: {b: dict(mean={c: wc["point"][m][k][j] for j, c in enumerate(cells)},
                          seed1={c: wc["per_seed"][m]["1"][k][j] for j, c in enumerate(cells)},
                          candidates=int(sum(wc["counts"][m]["1"][k])))
                  for k, b in enumerate(wc["order"]["bins"])} for m in ("CLOCs-R", "CLOCs-R_sel", "CLOCs-R_sub")}
    share_diff = {lvl: wc["bootstrap"][lvl] for lvl in ("scene", "recording")}

    # ---------------------------------------------------------------- verdicts (PREDICTIONS.md)
    v = {}
    c1 = comp("CLOCs-R_sel", "CLOCs-R_sub")["50-200"]
    v["P1"] = dict(test="CLOCs-R_sel - CLOCs-R_sub, val 50-200 m", **c1,
                   verdict=interval_verdict(c1["diff"], *c1["scene"], min_diff=0.005))
    x = share_diff["scene"]["CLOCs-R_sub - CLOCs-R_sel"]["50-100"]["camera"]
    xr = share_diff["recording"]["CLOCs-R_sub - CLOCs-R_sel"]["50-100"]["camera"]
    v["P2"] = dict(test="camera share at 50-100 m, CLOCs-R_sub - CLOCs-R_sel (6-seed means), val",
                   sub=shares["CLOCs-R_sub"]["50-100"]["mean"]["camera"], sel=shares["CLOCs-R_sel"]["50-100"]["mean"]["camera"],
                   diff=x["diff"], scene=[x["lo"], x["hi"]], recording=[xr["lo"], xr["hi"]],
                   verdict=interval_verdict(x["diff"], x["lo"], x["hi"]))
    c3 = comp("CLOCs-R_sub", "LC_sub")["50-200"]
    includes_zero = c3["scene"][0] <= 0 <= c3["scene"][1]
    v["P3"] = dict(test="CLOCs-R_sub - LC_sub, val 50-200 m: below +0.005, or the interval includes zero", **c3,
                   below_0_005=c3["diff"] < 0.005, interval_includes_zero=includes_zero,
                   verdict="supported" if (c3["diff"] < 0.005 or includes_zero) else "contradicted")
    c4 = comp("L0_sub", "L0_sel")["50-200"]
    v["P4"] = dict(test="L0_sub - L0_sel, val 50-200 m (no predicted direction)", **c4,
                   outcome=("L0_sub above L0_sel" if c4["scene"][0] > 0 else "L0_sub below L0_sel" if c4["scene"][1] < 0
                            else "interval includes zero"), verdict="no verdict (reported in either direction)")

    pairs = ([(f"{m}_sel", f"{m}_sub") for m in ("CLOCs-R", "LC", "L0", "LRd")] + [("L0_sub", "L0_sel")]
             + [("CLOCs-R_sub", "LC_sub"), ("LRd_sub", "L0_sub")] + [(f"{m}_sub", "L") for m in NETS]
             + [(f"{m}_sub", m) for m in NETS])
    diffs = {f"{a} - {b}": comp(a, b) for a, b in pairs}

    # ---------------------------------------------------------------- training: steps, frames and time
    t = json.load(open(S / "train_primary.json"))
    training = {k: dict(train_split=r.get("train_split"), train_scenes=r.get("train_scenes"), train_frames=r.get("train_frames"),
                        steps=r.get("steps"), last_logged_step={str(x["seed"]): x.get("steps") for x in r["runs"]})
                for k, r in t.items()}
    spans, starts = {}, {}
    for line in open(S / "times.tsv"):
        ts, msg = line.rstrip("\n").split("\t", 1)
        if msg.endswith(" start"):
            starts[msg[:-6]] = int(ts)
        elif msg.endswith(" done") and msg[:-5] in starts:
            spans[msg[:-5]] = int(ts) - starts[msg[:-5]]

    out = dict(subset=dict(draw=subset["draw"], scenes=subset["scenes"], recordings=subset["recordings"],
                           fit_sub=subset["fit_sub"], select=subset["select"], fit=subset["fit"]),
               verdicts=v, per_model=per_model, differences=diffs, winning_cells=dict(shares=shares, bootstrap=share_diff),
               training=dict(runs=training, wall_seconds=spans), reproduction=val["reproduction"],
               val_scoring_seconds=val["total_seconds"])
    json.dump(out, open(O / "results.json", "w"), indent=1)

    # ---------------------------------------------------------------- tables (markdown)
    L = ["### The subset: frames, recordings and far cars (scored cars, BEV range), against select and fit\n",
         "| Condition | fit_sub frames | select frames | fit_sub scenes / recordings | select scenes / recordings "
         "| fit_sub cars 50–100 / 100–200 m | select cars 50–100 / 100–200 m |", "|---|---|---|---|---|---|---|"]
    for c in CONDS + ["all"]:
        a, s = subset["fit_sub"][c], subset["select"][c]
        L.append(f"| {c} | {a['frames']:,} | {s['frames']:,} | {a['scenes']} / {a['recordings']} | {s['scenes']} / {s['recordings']} "
                 f"| {a['cars_50_100']:,} / {a['cars_100_200']:,} | {s['cars_50_100']:,} / {s['cars_100_200']:,} |")
    L.append("\n### Per model, val APCD (mean ± seed SD, seeds 1–6)\n")
    L.append("| Model | all | 0–30 | 30–50 | 50–100 | 100–200 | 50–200 |\n|---|---|---|---|---|---|---|")
    for m in order:
        row = [f3(x["mean"]) + ("" if x["sd"] is None else f" ± {f3(x['sd'])}") for x in (per_model[m]["distance"][b] for b in DIST)]
        L.append(f"| {label(m)} | " + " | ".join(row) + " |")
    L.append("\n### Per model, val APCD at 50–200 m by condition (mean ± seed SD)\n")
    L.append("| Model | highway | urban | night | rain |\n|---|---|---|---|---|")
    for m in order:
        row = [f3(x["mean"]) + ("" if x["sd"] is None else f" ± {f3(x['sd'])}")
               for x in (per_model[m]["condition_50_200"][c] for c in CONDS)]
        L.append(f"| {label(m)} | " + " | ".join(row) + " |")
    for bb, title in (("50-200", "50–200 m"), ("50-100", "50–100 m"), ("100-200", "100–200 m")):
        L.append(f"\n### Seed-paired differences on val at {title} (95% intervals)\n")
        L.append("| Difference | Δ | scene | recording |\n|---|---|---|---|")
        for k, d in diffs.items():
            a, b = k.split(" - ")
            x = d[bb]
            L.append(f"| {label(a)} − {label(b)} | {f3(x['diff'], True)} | [{f3(x['scene'][0], True)}, {f3(x['scene'][1], True)}] "
                     f"| [{f3(x['recording'][0], True)}, {f3(x['recording'][1], True)}] |")
    L.append("\n### Winning cell at SECOND candidates matched to val cars (share, %)\n")
    L.append("| Model | bin | candidates | camera | radar | reserved | camera, seed 1 |\n|---|---|---|---|---|---|---|")
    for m in ("CLOCs-R", "CLOCs-R_sel", "CLOCs-R_sub"):
        for b in ("50-100", "100-200"):
            x = shares[m][b]
            L.append(f"| {m} | {b} m | {x['candidates']:,} | " + " | ".join(f"{100 * x['mean'][c]:.1f}" for c in cells)
                     + f" | {100 * x['seed1']['camera']:.1f} |")
    L.append("\n| Share difference (points) | bin | Δ | scene | recording |\n|---|---|---|---|---|")
    for k in share_diff["scene"]:
        for b in ("50-100", "100-200"):
            for c in cells:
                x, r = share_diff["scene"][k][b][c], share_diff["recording"][k][b][c]
                a, bb = k.split(" - ")
                L.append(f"| {a} − {bb}, {c} | {b} m | {100 * x['diff']:+.1f} | [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}] "
                         f"| [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] |")
    (O / "report_tables.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: x["verdict"] if k != "P4" else x["outcome"] for k, x in v.items()}))


if __name__ == "__main__":
    sys.exit(main())
