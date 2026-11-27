"""Val, scored once for every _sub model of PREDICTIONS.md, after every training run has finished.

Reuses results/select_trained/score_val.py unchanged (its scoring job, winning cells, resampling and stored-value
reproduction); only the models, the comparisons and the share differences are this experiment's. The fit-trained
and select-trained networks are scored again from their saved weights in the same pass, so that every difference is
seed-paired and bootstrapped on the same resamples. They must reproduce their stored val results:
- fit-trained: results/lcr/data (as in results/select_trained);
- select-trained: results/select_trained/val.json (per-seed APCD, winning-cell counts, and the scene and recording
  intervals of the comparisons both files contain).

    python results/fit_sub_control/score_val.py --processes 5 --out results/fit_sub_control/val.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/select_trained"))
import score_val as sv  # noqa: E402  (results/select_trained/score_val.py)
from clocs_r import bootstrap as bs, make_qualitative as mq, splits  # noqa: E402

SUB = pathlib.Path("runs/fit_sub_control/primary")
SEL_VAL = pathlib.Path("results/select_trained/val.json")
NETS = ("L0", "LRd", "LC", "CLOCs-R")
CODE = {"L0": "R0+T2+NC", "LRd": "R0+T2+RP", "LC": "R0+T2+K40", "CLOCs-R": "R0+T2+K40+RC"}

MODELS = {"L": ("second", None, None),
          **{m: sv.FIT[m] for m in NETS},
          **{f"{m}_sel": ("fusion", CODE[m], str(sv.SEL / "primary")) for m in NETS},
          **{f"{m}_sub": ("fusion", CODE[m], str(SUB)) for m in NETS}}
COMPARISONS = (
    [(f"{m}_sel", f"{m}_sub") for m in ("CLOCs-R", "LC", "L0", "LRd")]  # P1, P4: select-trained minus fit_sub-trained
    + [("L0_sub", "L0_sel")]  # P4, in the direction the prompt states it
    + [("CLOCs-R_sub", "LC_sub"), ("LRd_sub", "L0_sub")]  # radar effects (P3)
    + [(f"{m}_sub", "L") for m in NETS]  # minus SECOND
    + [(f"{m}_sub", m) for m in NETS]  # minus the full-fit counterpart
    # reproduction only: comparisons also in results/lcr/data and results/select_trained/val.json
    + [("CLOCs-R", "LC"), ("CLOCs-R", "L"), ("LRd", "L0")]
    + [("CLOCs-R_sel", "LC_sel"), ("CLOCs-R_sel", "CLOCs-R"), ("L0_sel", "L0"), ("LRd_sel", "L0_sel"), ("CLOCs-R_sel", "L")]
)
SHARE_MODELS = ("CLOCs-R", "CLOCs-R_sel", "CLOCs-R_sub")
SHARE_COMPARISONS = [("CLOCs-R_sub", "CLOCs-R_sel"), ("CLOCs-R_sub", "CLOCs-R"),  # P2, and against fit-trained
                     ("CLOCs-R_sel", "CLOCs-R")]  # reproduction of results/select_trained/val.json


def reproduce_select(results, shares, boot, share_boot):
    ref = json.load(open(SEL_VAL))
    out = {}
    for m in NETS:
        lab = f"{m}_sel"
        if lab not in results:
            continue
        diffs = [abs(results[lab][s][g][b]["APCD"] - ref["results"][lab][str(s)][g][b]["APCD"])
                 for s in results[lab] for g in ref["results"][lab][str(s)] if g in results[lab][s]
                 for b in ref["results"][lab][str(s)][g]]
        out[lab] = dict(seeds=sorted(results[lab]), values_compared=len(diffs), max_abs_diff=max(diffs))
    if "CLOCs-R_sel" in shares:
        out["winning_cell_counts_CLOCs-R_sel"] = dict(identical=all(
            shares["CLOCs-R_sel"][s].sum(0).tolist() == ref["winning_cells"]["counts"]["CLOCs-R_sel"][str(s)]
            for s in shares["CLOCs-R_sel"]))
    for level in ("scene", "recording"):
        for k in boot[level]:
            if k in ref["bootstrap"][level] and ("_sel" in k or k.startswith("CLOCs-R - ") or k.startswith("LRd - ")):
                d = max(abs(boot[level][k][b][x] - ref["bootstrap"][level][k][b][x]) for b in sv.BINS for x in ("diff", "lo", "hi"))
                out.setdefault(f"bootstrap_{level}", {})[k] = dict(max_abs_diff=float(d))
        k = "CLOCs-R_sel - CLOCs-R"
        if k in share_boot[level] and k in ref["winning_cells"]["bootstrap"][level]:
            r = ref["winning_cells"]["bootstrap"][level][k]
            d = max(abs(share_boot[level][k][b][c][x] - r[b][c][x]) for b in r for c in r[b] for x in ("diff", "lo", "hi"))
            out.setdefault(f"share_bootstrap_{level}", {})[k] = dict(max_abs_diff=float(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", type=int, default=3)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="tests only: the first N val frames")
    ap.add_argument("--models", default="", help="tests only: comma-separated labels (default: all)")
    ap.add_argument("--seeds", default="", help="tests only: comma-separated seeds (default: 1-6)")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(sv.SEEDS)
    models = dict(MODELS)
    comps = list(COMPARISONS)
    if args.models:
        keep = args.models.split(",")
        models = {k: v for k, v in models.items() if k in keep}
        comps = [c for c in comps if c[0] in models and c[1] in models]
    sv.MODELS.clear()
    sv.MODELS.update(models)  # the scoring job of results/select_trained reads its module's MODELS
    t0 = time.time()
    sv.D["val"] = bs.load_split("val", sv.R / "cells", sv.R / "cand/3d", sv.R / "cand/2d", sv.R / "d2_dets")
    if args.limit:
        sv.D["val"] = bs.subset(sv.D["val"], np.arange(args.limit))
    sv.D["matched"] = sv.matched_candidates(sv.D["val"])
    items = sv.D["val"]["items"]
    jobs = [("L", 0)] + [(m, s) for m in models if m != "L" for s in seeds]
    results, matches, shares, seconds = {}, {}, {}, {}
    counter = mp.get_context("fork").Value("i", 0)
    with mp.get_context("fork").Pool(args.processes, initializer=sv._init, initargs=(counter,)) as pool:
        for label, seed, res, m, share, sec in pool.imap_unordered(sv.job, jobs):
            results.setdefault(label, {})[seed] = res
            matches[(label, seed)] = m
            if share is not None:
                shares.setdefault(label, {})[seed] = share
            seconds[f"{label} s{seed}"] = round(sec, 1)
            print(f"scored {label} s{seed} ({sec:.0f} s; {time.time() - t0:.0f} s total)", flush=True)

    scenes = sorted({it["seq"] for it in items})
    recordings = sorted({splits.recording(s) for s in scenes})
    unit = {"scene": np.array([scenes.index(it["seq"]) for it in items]),
            "recording": np.array([recordings.index(splits.recording(it["seq"])) for it in items])}

    def value(label, w, b):
        return float(np.mean([bs.weighted_apcd(matches[k][b], w) for k in matches if k[0] == label]))

    ones = np.ones(len(items))
    point = {label: {b: value(label, ones, b) for b in sv.BINS} for label in models}
    boot = {}
    for level, u in unit.items():
        def draw(w):
            cache = {}
            for a, c in comps:
                for lab in (a, c):
                    for b in sv.BINS:
                        if (lab, b) not in cache:
                            cache[(lab, b)] = value(lab, w, b)
            return {f"{a} - {c}": {b: cache[(a, b)] - cache[(c, b)] for b in sv.BINS} for a, c in comps}
        dr = sv.resample(u, int(u.max()) + 1, draw, args.resamples)
        boot[level] = {}
        for a, c in comps:
            k = f"{a} - {c}"
            boot[level][k] = {b: dict(diff=point[a][b] - point[c][b], lo=float(np.nanpercentile([x[k][b] for x in dr], 2.5)),
                                      hi=float(np.nanpercentile([x[k][b] for x in dr], 97.5)),
                                      p_le_0=float(np.mean(np.array([x[k][b] for x in dr]) <= 0)),
                                      nan_draws=int(np.isnan([x[k][b] for x in dr]).sum())) for b in sv.BINS}
        print(f"{level} bootstrap done ({time.time() - t0:.0f} s)", flush=True)

    # winning-cell shares: per seed from the (weighted) counts, then the mean of seeds (results/select_trained)
    def share_of(label, w):
        per = []
        for seed, c in shares[label].items():
            t = (c * w[:, None, None]).sum(0)  # (2 bins, 3 types)
            per.append(t / t.sum(1, keepdims=True))
        return np.mean(per, 0)
    share_models = [m for m in SHARE_MODELS if m in shares]
    share_comps = [(a, c) for a, c in SHARE_COMPARISONS if a in share_models and c in share_models]
    share_point = {label: share_of(label, ones) for label in share_models}
    share_seed = {label: {s: (c.sum(0) / c.sum(0).sum(1, keepdims=True)).tolist() for s, c in shares[label].items()}
                  for label in share_models}
    share_boot = {}
    for level, u in unit.items():
        dr = sv.resample(u, int(u.max()) + 1, lambda w: {lab: share_of(lab, w) for lab in share_models}, args.resamples)
        share_boot[level] = {}
        for a, c in share_comps:
            diffs = np.array([x[a] - x[c] for x in dr])  # (resamples, 2, 3)
            share_boot[level][f"{a} - {c}"] = {
                b: {n_: dict(diff=float(share_point[a][k, j] - share_point[c][k, j]),
                             lo=float(np.percentile(diffs[:, k, j], 2.5)), hi=float(np.percentile(diffs[:, k, j], 97.5)))
                    for j, n_ in mq.CELL_NAMES.items()} for k, b in enumerate(("50-100", "100-200"))}
    print(f"share bootstrap done ({time.time() - t0:.0f} s)", flush=True)

    repro = sv.reproduction(results, shares, boot)  # fit-trained against results/lcr/data and results/qualitative
    for k in repro.get("bootstrap_scene", {}):
        for b in sv.BINS:
            h = boot["scene"][k][b]
            repro["bootstrap_scene"][k][b]["here"] = [h["diff"], h["lo"], h["hi"]]
            repro["bootstrap_scene"][k][b]["max_abs_diff"] = float(np.max(np.abs(np.subtract(
                repro["bootstrap_scene"][k][b]["here"], repro["bootstrap_scene"][k][b]["stored"]))))
    repro["select_trained"] = reproduce_select(results, shares, boot, share_boot)
    out = dict(split="val", frames=len(items), scenes=len(scenes),
               recordings={r: sum(splits.recording(s) == r for s in scenes) for r in recordings},
               resamples=args.resamples, models={k: [v[0], str(v[1]), str(v[2])] for k, v in models.items()},
               results={k: {str(s): {g: {b: {m: r[g][b][m] for m in ("APCD", "n_gt", "n_det")} for b in r[g]} for g in r}
                            for s, r in v.items()} for k, v in results.items()}, point=point, bootstrap=boot,
               winning_cells=dict(point={k: v.tolist() for k, v in share_point.items()}, per_seed=share_seed,
                                  counts={k: {str(s): c.sum(0).tolist() for s, c in v.items()} for k, v in shares.items()},
                                  bootstrap=share_boot, order=dict(bins=["50-100", "100-200"], cells=list(mq.CELL_NAMES.values()))),
               reproduction=repro, seconds=seconds, total_seconds=round(time.time() - t0, 1))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(repro, indent=1))
    print("wrote", args.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
