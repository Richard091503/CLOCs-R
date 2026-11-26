"""results.json and the tables of REPORT.md, from the result files of results/select_trained (every number is read, none is
typed in). The verdicts follow PREDICTIONS.md's rules.

    python results/select_trained/make_report.py      # writes results.json and report_tables.md
"""
import json
import pathlib
import sys

import numpy as np
import torch

O = pathlib.Path("results/select_trained")
S = pathlib.Path("runs/select_trained")
THESIS = pathlib.Path("../CLOCs-R-thesis")
SEEDS = ["1", "2", "3", "4", "5", "6"]
DIST = ["all", "0-30", "30-50", "50-100", "100-200", "50-200"]
CONDS = ["highway", "urban", "night", "rain"]
MODELS = ["L0", "LRd", "LC", "CLOCs-R", "B"]
NAME = {"L": "SECOND", "L0": "L0", "LRd": "LRd", "LC": "LC", "CLOCs-R": "CLOCs-R", "B": "B"}


def label(m):
    base, _, suf = m.partition("_")
    return NAME.get(base, base) + ("" if not suf else "_sel" if suf == "sel" else "_sel (3 ep.)")


def f3(x, sign=False):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:+.3f}" if sign else f"{x:.3f}")


def seed_values(val, m, group, b):
    per = val["results"][m]
    return np.array([per[s][group][b]["APCD"] for s in (SEEDS if m != "L" else ["0"]) if group in per[s]])


def mean_sd(v):
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None, seeds=v.tolist())


def verdict_interval(diff, lo, hi, min_diff=None):
    if hi < 0:
        return "contradicted"
    if lo > 0 and (min_diff is None or diff >= min_diff):
        return "supported"
    return "not supported"


def main():
    val = json.load(open(O / "val.json"))
    step0 = json.load(open(O / "step0.json"))
    equiv = json.load(open(O / "equivalence.json"))
    boot = val["bootstrap"]

    def comp(a, b, bins=("50-200", "100-200", "50-100")):
        k = f"{a} - {b}"
        return {bb: dict(diff=boot["scene"][k][bb]["diff"], scene=[boot["scene"][k][bb]["lo"], boot["scene"][k][bb]["hi"]],
                         recording=[boot["recording"][k][bb]["lo"], boot["recording"][k][bb]["hi"]]) for bb in bins}

    # ---------------------------------------------------------------- per-model tables
    per_model = {}
    for m in ["L"] + MODELS + [f"{x}_sel" for x in MODELS] + [f"{x}_sel3" for x in MODELS]:
        per_model[m] = dict(distance={b: mean_sd(seed_values(val, m, "overall", b)) for b in DIST},
                            condition_50_200={c: mean_sd(seed_values(val, m, c, "50-200")) for c in CONDS})

    # ---------------------------------------------------------------- winning cells
    wc = val["winning_cells"]
    cells = wc["order"]["cells"]
    shares = {}
    for m in ("CLOCs-R", "CLOCs-R_sel", "CLOCs-R_sel3"):
        shares[m] = {b: dict(mean={c: wc["point"][m][k][j] for j, c in enumerate(cells)},
                             seed1={c: wc["per_seed"][m]["1"][k][j] for j, c in enumerate(cells)},
                             candidates=int(sum(wc["counts"][m]["1"][k])))
                     for k, b in enumerate(wc["order"]["bins"])}
    share_diff = {lvl: wc["bootstrap"][lvl] for lvl in ("scene", "recording")}

    # ---------------------------------------------------------------- verdicts
    h = {}
    c1 = comp("CLOCs-R_sel", "LC_sel")["50-200"]
    h["H1"] = dict(test="CLOCs-R_sel - LC_sel, val 50-200 m", **c1, fit_trained=comp("CLOCs-R", "LC")["50-200"],
                   verdict=verdict_interval(c1["diff"], *c1["scene"], min_diff=0.005))
    cam = {b: share_diff["scene"]["CLOCs-R_sel - CLOCs-R"][b]["camera"] for b in ("50-100", "100-200")}
    cam_rec = {b: share_diff["recording"]["CLOCs-R_sel - CLOCs-R"][b]["camera"] for b in ("50-100", "100-200")}
    if all(v["diff"] < 0 and v["hi"] < 0 for v in cam.values()):
        v2 = "supported"
    elif any(v["diff"] > 0 and v["lo"] > 0 for v in cam.values()):
        v2 = "contradicted"
    else:
        v2 = "not supported"
    h["H2"] = dict(test="camera share of CLOCs-R_sel - fit-trained CLOCs-R (6-seed means), val",
                   **{b: dict(sel=shares["CLOCs-R_sel"][b]["mean"]["camera"], fit=shares["CLOCs-R"][b]["mean"]["camera"],
                              fit_seed1=shares["CLOCs-R"][b]["seed1"]["camera"], diff=cam[b]["diff"],
                              scene=[cam[b]["lo"], cam[b]["hi"]], recording=[cam_rec[b]["lo"], cam_rec[b]["hi"]])
                      for b in cam}, verdict=v2)
    c3 = comp("CLOCs-R_sel", "CLOCs-R")["50-200"]
    vs_l = comp("CLOCs-R_sel", "L")["50-200"]
    h["H3"] = dict(test="CLOCs-R_sel - CLOCs-R (fit-trained), val 50-200 m (= the change in the gap to SECOND)", **c3,
                   sel_minus_second=vs_l, fit_minus_second=comp("CLOCs-R", "L")["50-200"],
                   sel_minus_second_interval_excludes_zero=("above" if vs_l["scene"][0] > 0 else "below" if vs_l["scene"][1] < 0 else "no"),
                   verdict=verdict_interval(c3["diff"], *c3["scene"]))
    c4 = comp("L0_sel", "L0")["50-200"]
    l0s, l0f = comp("L0_sel", "L")["50-200"], comp("L0", "L")["50-200"]
    closer = abs(l0s["diff"]) < abs(l0f["diff"])
    v4 = "contradicted" if c4["scene"][1] < 0 else "supported" if (closer and c4["scene"][0] > 0) else "not supported"
    h["H4"] = dict(test="L0_sel - L0 (fit-trained), val 50-200 m; supported needs |L0_sel - SECOND| < |L0 - SECOND|", **c4,
                   sel_minus_second=l0s, fit_minus_second=l0f, closer_to_zero=closer, verdict=v4)
    sec = {}
    s1 = comp("CLOCs-R_sel3", "LC_sel3")["50-200"]
    sec["H1"] = dict(**s1, verdict=verdict_interval(s1["diff"], *s1["scene"], min_diff=0.005))
    s3 = comp("CLOCs-R_sel3", "CLOCs-R")["50-200"]
    sec["H3"] = dict(**s3, sel_minus_second=comp("CLOCs-R_sel3", "L")["50-200"], verdict=verdict_interval(s3["diff"], *s3["scene"]))

    # ---------------------------------------------------------------- differences
    diffs = {}
    for a, b in ([(f"{m}_sel", m) for m in MODELS] + [("CLOCs-R_sel", "LC_sel"), ("LRd_sel", "L0_sel"), ("LC_sel", "L0_sel"),
                                                      ("CLOCs-R_sel", "LRd_sel")] + [(f"{m}_sel", "L") for m in MODELS]
                 + [("B_sel", "CLOCs-R_sel"), ("CLOCs-R_sel3", "LC_sel3"), ("CLOCs-R_sel3", "CLOCs-R"), ("CLOCs-R_sel3", "L")]):
        diffs[f"{a} - {b}"] = comp(a, b)

    # ---------------------------------------------------------------- B weights
    weights = {}
    for length in ("primary", "secondary"):
        runs = json.load(open(S / f"train_{length}_blend.json"))[f"B_sel {length}"]["runs"]
        weights[f"B_sel {length}"] = dict(per_seed={str(r["seed"]): r["weights"] for r in runs},
                                          mean={e: float(np.mean([r["weights"][e] for r in runs])) for e in ("L", "LC", "LRd")},
                                          blend_steps=runs[0]["blend_steps"], oof=runs[0]["oof"])
    fitw = {}
    for s in SEEDS:
        st = torch.load(THESIS / f"results/next_round/phase7/blend_R0+T2+RP_s{s}.pth", map_location="cpu")
        fitw[s] = dict(zip(("L", "LC", "LRd"), torch.softmax(st["model"]["bias"], 0).tolist()))
    weights["B (fit-trained)"] = dict(per_seed=fitw, mean={e: float(np.mean([w[e] for w in fitw.values()])) for e in ("L", "LC", "LRd")})
    folds = json.load(open(S / "train_primary_blend.json"))["B_sel primary"]

    # ---------------------------------------------------------------- training: steps and time
    steps = {}
    for length in ("primary", "secondary"):
        t = json.load(open(S / f"train_{length}.json"))
        steps[length] = {k: dict(train_split=v.get("train_split"), steps=v.get("steps"),
                                 per_seed={str(r["seed"]): r.get("steps") for r in v["runs"]}) for k, v in t.items()}
    times = []
    for line in open(S / "times.tsv"):
        t, msg = line.rstrip("\n").split("\t", 1)
        times.append((int(t), msg))
    spans = {}
    starts = {}
    for t, msg in times:
        if msg.endswith(" start"):
            starts[msg[:-6]] = t
        elif msg.endswith(" done") and msg[:-5] in starts:
            spans[msg[:-5]] = t - starts[msg[:-5]]
        elif msg.startswith("step 0 done") and "step 0" in starts:
            spans["step 0"] = t - starts["step 0"]
    blend_sec = {length: {str(r["seed"]): r["seconds"] for r in json.load(open(S / f"train_{length}_blend.json"))[f"B_sel {length}"]["runs"]}
                 for length in ("primary", "secondary")}

    out = dict(step0=step0, equivalence=equiv, verdicts=h, secondary_verdicts=sec, per_model=per_model, differences=diffs,
               winning_cells=dict(shares=shares, bootstrap=share_diff), blend_weights=weights,
               blend_folds=dict(folds=folds["folds"], fold_frames=folds["fold_frames"]),
               training=dict(steps=steps, wall_seconds=spans, blend_seconds_per_seed=blend_sec),
               reproduction=val["reproduction"], val_scoring_seconds=val["total_seconds"])
    json.dump(out, open(O / "results.json", "w"), indent=1)

    # ---------------------------------------------------------------- tables (markdown)
    L = []
    L.append("### Per model, val APCD (mean ± seed SD, seeds 1–6)\n")
    L.append("| Model | all | 0–30 | 30–50 | 50–100 | 100–200 | 50–200 |\n|---|---|---|---|---|---|---|")
    for m in ["L"] + [x for p in MODELS for x in (f"{p}_sel", p)]:
        row = []
        for b in DIST:
            v = per_model[m]["distance"][b]
            row.append(f3(v["mean"]) + ("" if v["sd"] is None else f" ± {f3(v['sd'])}"))
        L.append(f"| {label(m)} |" + " | ".join(row) + " |")
    L.append("\n### Per model, val APCD at 50–200 m by condition (mean ± seed SD)\n")
    L.append("| Model | highway | urban | night | rain |\n|---|---|---|---|---|")
    for m in ["L"] + [x for p in MODELS for x in (f"{p}_sel", p)]:
        row = []
        for c in CONDS:
            v = per_model[m]["condition_50_200"][c]
            row.append(f3(v["mean"]) + ("" if v["sd"] is None else f" ± {f3(v['sd'])}"))
        L.append(f"| {label(m)} |" + " | ".join(row) + " |")
    L.append("\n### Seed-paired differences on val (scene and recording 95% intervals)\n")
    L.append("| Difference | 50–200 m | scene | recording | 100–200 m | scene | recording |\n|---|---|---|---|---|---|---|")
    for k, v in diffs.items():
        a, b = k.split(" - ")
        cells_ = []
        for bb in ("50-200", "100-200"):
            x = v[bb]
            cells_ += [f3(x["diff"], True), f"[{f3(x['scene'][0], True)}, {f3(x['scene'][1], True)}]",
                       f"[{f3(x['recording'][0], True)}, {f3(x['recording'][1], True)}]"]
        L.append(f"| {label(a)} − {label(b)} | " + " | ".join(cells_) + " |")
    L.append("\n### Winning cell at SECOND candidates matched to val cars (share, %)\n")
    L.append("| Model | bin | candidates | camera | radar | reserved | camera, seed 1 |\n|---|---|---|---|---|---|---|")
    for m in ("CLOCs-R", "CLOCs-R_sel", "CLOCs-R_sel3"):
        for b in ("50-100", "100-200"):
            x = shares[m][b]
            L.append(f"| {label(m)} | {b} m | {x['candidates']:,} | " + " | ".join(f"{100 * x['mean'][c]:.1f}" for c in cells)
                     + f" | {100 * x['seed1']['camera']:.1f} |")
    L.append("\n| Share difference (camera / radar / reserved, points) | bin | difference | scene | recording |\n|---|---|---|---|---|")
    for k in share_diff["scene"]:
        for b in ("50-100", "100-200"):
            for c in cells:
                x, r = share_diff["scene"][k][b][c], share_diff["recording"][k][b][c]
                a, bb = k.split(" - ")
                L.append(f"| {label(a)} − {label(bb)}, {c} | {b} m | {100 * x['diff']:+.1f} | [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}] "
                         f"| [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] |")
    L.append("\n### Blend weights (softmax, L / LC / LRd)\n")
    L.append("| Blend | seed 1 | 2 | 3 | 4 | 5 | 6 | mean |\n|---|---|---|---|---|---|---|---|")
    for k, v in weights.items():
        fmt = lambda w: " / ".join(f"{w[e]:.3f}" for e in ("L", "LC", "LRd"))  # noqa: E731
        L.append(f"| {k} | " + " | ".join(fmt(v["per_seed"][s]) for s in SEEDS) + f" | {fmt(v['mean'])} |")
    L.append("\n### Secondary (3 epochs over select): val APCD at 50–200 m\n")
    L.append("| Model | mean ± SD | primary length | fit-trained |\n|---|---|---|---|")
    for m in MODELS:
        a, p, f = (per_model[f"{m}_sel3"]["distance"]["50-200"], per_model[f"{m}_sel"]["distance"]["50-200"],
                   per_model[m]["distance"]["50-200"])
        L.append(f"| {m}_sel | {f3(a['mean'])} ± {f3(a['sd'])} | {f3(p['mean'])} ± {f3(p['sd'])} | {f3(f['mean'])} ± {f3(f['sd'])} |")
    (O / "report_tables.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: v["verdict"] for k, v in h.items()}), json.dumps({k: v["verdict"] for k, v in sec.items()}))


if __name__ == "__main__":
    sys.exit(main())
