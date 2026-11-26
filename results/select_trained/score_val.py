"""Val, scored once for every model of PREDICTIONS.md, after every training run has finished.

For each model and seed: the paper's APCD (metrics.evaluate, BEV NMS at 0.1, all distance bins, every driving
condition), the per-frame matches of clocs_r/bootstrap.py, and for CLOCs-R models the winning cell of each SECOND
candidate matched to a car at 50-200 m (camera, radar or reserved; results/qualitative's definition). Then:
- reproduction: the fit-trained models from their saved networks against the stored per-seed val results, and
  fit-trained CLOCs-R seed 1's winning-cell counts against results/qualitative/events.json;
- bootstrap: 1,000 resamples of scenes and, separately, of recordings (results/review_round/cluster_bootstrap.py:
  RandomState(0) per level, seeds averaged inside each resample, percentile 95% intervals);
- the same two bootstraps for the camera cell's share.

    python results/select_trained/score_val.py --out results/select_trained/val.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import blend  # noqa: E402
from clocs_r import aimotive, bootstrap as bs, make_qualitative as mq, metrics, splits, train_fusion  # noqa: E402
from clocs_r import tune_fusion as tf  # noqa: E402

R = pathlib.Path("runs/lcr")
STORED = pathlib.Path("results/lcr/data")
THESIS = pathlib.Path("../CLOCs-R-thesis")  # the thesis worktree beside this checkout: B's fit-trained weights
SEL = pathlib.Path("runs/select_trained")
SEEDS = [1, 2, 3, 4, 5, 6]
BINS = {b: bs.BINS[b] for b in ("50-100", "100-200", "50-200")}  # the bootstrapped bins
FIT = {  # the paper's fit-trained networks (the links in runs/lcr/fusion point to the same files)
    "L0": ("fusion", "R0+T2+NC", "runs/next_round/phase7/fusion"),
    "LRd": ("fusion", "R0+T2+RP", "runs/next_round/phase7/fusion_rp"),
    "LC": ("fusion", "R0+T2+K40", "runs/next_round/phase3/fusion"),
    "CLOCs-R": ("fusion", "R0+T2+K40+RC", "runs/next_round/phase7/fusion_lcr"),
    "B": ("blend", str(THESIS / "results/next_round/phase7/blend_R0+T2+RP_s{seed}.pth"),
          ("runs/next_round/phase3/fusion", "runs/next_round/phase7/fusion_rp")),
}


def sel_models(length, suffix):
    d = SEL / length
    return {"L0" + suffix: ("fusion", "R0+T2+NC", str(d)), "LRd" + suffix: ("fusion", "R0+T2+RP", str(d)),
            "LC" + suffix: ("fusion", "R0+T2+K40", str(d)), "CLOCs-R" + suffix: ("fusion", "R0+T2+K40+RC", str(d)),
            "B" + suffix: ("blend", str(d / "blend_s{seed}.pth"), (str(d), str(d)))}


MODELS = {"L": ("second", None, None), **FIT, **sel_models("primary", "_sel"), **sel_models("secondary", "_sel3")}
COMPARISONS = (
    [(f"{m}_sel", m) for m in ("L0", "LRd", "LC", "CLOCs-R", "B")]  # select-trained minus fit-trained
    + [("CLOCs-R_sel", "LC_sel"), ("LRd_sel", "L0_sel")]  # radar effects
    + [("LC_sel", "L0_sel"), ("CLOCs-R_sel", "LRd_sel")]  # camera effects
    + [(f"{m}_sel", "L") for m in ("L0", "LRd", "LC", "CLOCs-R", "B")]  # minus SECOND
    + [("B_sel", "CLOCs-R_sel")]
    + [("CLOCs-R", "LC"), ("CLOCs-R", "L"), ("L0", "L"), ("LRd", "L0"), ("B", "LC")]  # fit-trained, for the reproduction
    + [("CLOCs-R_sel3", "LC_sel3"), ("CLOCs-R_sel3", "CLOCs-R"), ("CLOCs-R_sel3", "L")]  # secondary: H1 and H3
)
SHARE_MODELS = ("CLOCs-R", "CLOCs-R_sel", "CLOCs-R_sel3")
D = {}


def matched_candidates(d):
    """Per frame: (candidate index, bin 0 = 50-100 m / 1 = 100-200 m) of each car at 50-200 m with a SECOND candidate
    within 2 m, matched greedily in descending SECOND score (make_qualitative.frame_cars, over the scored candidates)."""
    out = []
    none = (np.zeros((0, 7), np.float32), np.zeros(0, np.float32))
    for it in d["items"]:
        m = it["keep_eval"]
        idx = np.nonzero(m)[0]
        f = dict(gt=it["gt"], boxes=it["boxes"][m], s={"SECOND": it["scores"][m]})
        rows = [r for r in mq.frame_cars(f, none) if r["cand"] >= 0]
        cand = np.array([idx[r["cand"]] for r in rows], np.int64)
        dist = np.array([np.hypot(*it["gt"][r["g"], :2]) for r in rows])
        out.append((cand, (dist >= 100).astype(np.int64)))
    return out


@torch.no_grad()
def winning_cells(net, d, device):
    """(frames, 2 bins, 3 cell types) counts of the winning cell at the matched candidates (make_qualitative.score_split)."""
    out = np.zeros((len(d["rc"]), 2, 3), np.int64)
    for i, it in enumerate(d["rc"]):
        cand, b = D["matched"][i]
        if not len(cand):
            continue
        x, col = train_fusion.frame_cells(it, device)
        z = torch.logit(x[:, 2].clamp(1e-6, 1 - 1e-6))
        cell = (net.net(x.t()[None, :, None, :])[0, 0, 0] + z).cpu().numpy()
        c = col.cpu().numpy()
        order = np.lexsort((cell, c))
        sc = c[order]
        last = np.r_[sc[1:] != sc[:-1], True]
        win = np.full(len(it["scores"]), -1, np.int64)
        win[sc[last]] = order[last]
        kind = mq.cell_type(it["pairs"][win[cand]].astype(np.float32))
        np.add.at(out[i], (b, kind), 1)
    return out


def job(args):
    label, seed = args
    torch.set_num_threads(1)
    t0 = time.time()
    d = D["val"]
    kind, a, b = MODELS[label]
    share = None
    if kind == "second":
        sc = [it["scores"] for it in d["items"]]
    else:
        device = torch.device("cuda")
        if kind == "fusion":
            sc = bs.fusion_scores(a, b, seed, d, device)
            if tf.parse(a).get("rc"):
                share = winning_cells(tf.load_net(tf.parse(a), a, b, seed, 2.0, device).eval(), d, device)
        else:
            gate = blend.load_blend(a.format(seed=seed), device)
            sc = blend.blend_scores(gate, d, b[0], b[1], seed, device)
    kept = bs.kept_frames(d["items"], sc)
    records = [(it["gt"], k[1], k[2], it["odd"], it["seq"]) for it, k in zip(d["items"], kept)]
    res = {name: metrics.evaluate([r[:3] for r in rows], bins=aimotive.DIST_BINS + [train_fusion.FAR])
           for name, rows in train_fusion.condition_groups(records).items()}
    m = bs.frame_matches(kept)
    return label, seed, res, {k: m[k] for k in BINS}, share, time.time() - t0


def _init(counter):
    with counter.get_lock():
        k = counter.value
        counter.value += 1
    time.sleep(tf.STAGGER_S * k)  # one CUDA start-up at a time (tune_fusion.STAGGER_S)


def stored_val():
    def seeds(files, name):
        return {r["seed"]: r["val"] for f in files for r in json.load(open(STORED / f)).get(name, {}).get("runs", [])}
    return {"L": {0: json.load(open(STORED / "aimotive_second.json"))["val"]},
            "L0": seeds(["val_experts_nc.json"], "R0+T2+NC"), "LRd": seeds(["val_experts_rp.json"], "R0+T2+RP"),
            "LC": seeds(["final_r0_s123.json", "final_s456.json"], "R0+T2+K40"),
            "CLOCs-R": seeds(["lcr_val.json"], "R0+T2+K40+RC"), "B": seeds(["blend_val.json"], "B(R0+T2+RP)")}


def reproduction(results, shares, boot):
    st = stored_val()
    out = {}
    for label, per in st.items():
        if label not in results:  # tests with --models only
            continue
        diffs = []
        for seed, ref in per.items():
            if seed not in results[label]:  # tests with --seeds only
                continue
            here = results[label][seed]
            for group in ref:
                if group in here:
                    for b in ref[group]:
                        diffs.append(abs(here[group][b]["APCD"] - ref[group][b]["APCD"]))
        out[label] = dict(seeds=sorted(per), values_compared=len(diffs), max_abs_diff=max(diffs))
    if "CLOCs-R" not in shares or 1 not in shares["CLOCs-R"]:
        return out
    ev = json.load(open("results/qualitative/events.json"))["summary"]["val"]["winning_cell_share"]
    s1 = shares["CLOCs-R"][1].sum(0)
    q = {}
    for k, b in enumerate(("50-100", "100-200")):
        q[b] = dict(here=dict(candidates=int(s1[k].sum()), **{n: float(s1[k, j] / s1[k].sum()) for j, n in mq.CELL_NAMES.items()}),
                    stored={n: ev[b][n] for n in ("candidates", "camera", "radar", "reserved")})
        q[b]["identical"] = q[b]["here"] == q[b]["stored"]
    out["winning_cells_CLOCs-R_s1"] = q
    for f, comps in (("bootstrap_lcr_val.json", {"CLOCs-R - LC": "LCR - LC", "CLOCs-R - L": "LCR - L"}),
                     ("bootstrap_val.json", {"LRd - L0": "LRd - L0", "B - LC": "B - LC"})):
        ref = json.load(open(STORED / f))["comparisons"]
        for here, there in comps.items():
            if here not in boot["scene"]:
                continue
            out.setdefault("bootstrap_scene", {})[here] = {b: dict(stored=[ref[there][b][k] for k in ("diff", "lo", "hi")])
                                                           for b in BINS}
    return out


def resample(units, n_units, draws_fn, resamples=1000):
    rng = np.random.RandomState(0)
    out = []
    for _ in range(resamples):
        w = np.bincount(rng.randint(0, n_units, n_units), minlength=n_units)[units].astype(np.float64)
        out.append(draws_fn(w))
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
    if args.seeds:
        SEEDS[:] = [int(x) for x in args.seeds.split(",")]
    if args.models:
        keep = args.models.split(",")
        for k in [k for k in MODELS if k not in keep]:
            del MODELS[k]
        COMPARISONS[:] = [c for c in COMPARISONS if c[0] in MODELS and c[1] in MODELS]
    t0 = time.time()
    D["val"] = bs.load_split("val", R / "cells", R / "cand/3d", R / "cand/2d", R / "d2_dets")
    if args.limit:
        D["val"] = bs.subset(D["val"], np.arange(args.limit))
    D["matched"] = matched_candidates(D["val"])
    items = D["val"]["items"]
    jobs = [("L", 0)] + [(m, s) for m in MODELS if m != "L" for s in SEEDS]
    results, matches, shares, seconds = {}, {}, {}, {}
    counter = mp.get_context("fork").Value("i", 0)
    with mp.get_context("fork").Pool(args.processes, initializer=_init, initargs=(counter,)) as pool:
        for label, seed, res, m, share, sec in pool.imap_unordered(job, jobs):
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
    point = {label: {b: value(label, ones, b) for b in BINS} for label in MODELS}
    boot = {}
    for level, u in unit.items():
        def draw(w):
            cache = {}
            for a, c in COMPARISONS:
                for lab in (a, c):
                    for b in BINS:
                        if (lab, b) not in cache:
                            cache[(lab, b)] = value(lab, w, b)
            return {f"{a} - {c}": {b: cache[(a, b)] - cache[(c, b)] for b in BINS} for a, c in COMPARISONS}
        dr = resample(u, int(u.max()) + 1, draw, args.resamples)
        boot[level] = {}
        for a, c in COMPARISONS:
            k = f"{a} - {c}"
            boot[level][k] = {b: dict(diff=point[a][b] - point[c][b], lo=float(np.nanpercentile([x[k][b] for x in dr], 2.5)),
                                      hi=float(np.nanpercentile([x[k][b] for x in dr], 97.5)),
                                      p_le_0=float(np.mean(np.array([x[k][b] for x in dr]) <= 0)),
                                      nan_draws=int(np.isnan([x[k][b] for x in dr]).sum())) for b in BINS}
        print(f"{level} bootstrap done ({time.time() - t0:.0f} s)", flush=True)

    # camera cell's share of the matched candidates: per seed, then the mean of seeds; the difference select - fit
    def share_of(label, w):
        per = []
        for seed, c in shares[label].items():
            t = (c * w[:, None, None]).sum(0)  # (2 bins, 3 types)
            per.append(t / t.sum(1, keepdims=True))
        return np.mean(per, 0)
    share_models = [m for m in SHARE_MODELS if m in shares]
    share_point = {label: share_of(label, ones) for label in share_models}
    share_seed = {label: {s: (c.sum(0) / c.sum(0).sum(1, keepdims=True)).tolist() for s, c in shares[label].items()}
                  for label in share_models}
    share_boot = {}
    for level, u in unit.items():
        n = int(u.max()) + 1
        dr = resample(u, n, lambda w: {lab: share_of(lab, w) for lab in share_models}, args.resamples)
        share_boot[level] = {}
        for a in [m for m in ("CLOCs-R_sel", "CLOCs-R_sel3") if m in share_models and "CLOCs-R" in share_models]:
            diffs = np.array([x[a] - x["CLOCs-R"] for x in dr])  # (resamples, 2, 3)
            share_boot[level][f"{a} - CLOCs-R"] = {
                b: {n_: dict(diff=float(share_point[a][k, j] - share_point["CLOCs-R"][k, j]),
                             lo=float(np.percentile(diffs[:, k, j], 2.5)), hi=float(np.percentile(diffs[:, k, j], 97.5)))
                    for j, n_ in mq.CELL_NAMES.items()} for k, b in enumerate(("50-100", "100-200"))}
    print(f"share bootstrap done ({time.time() - t0:.0f} s)", flush=True)

    repro = reproduction(results, shares, boot)
    for k in repro.get("bootstrap_scene", {}):
        for b in BINS:
            h = boot["scene"][k][b]
            repro["bootstrap_scene"][k][b]["here"] = [h["diff"], h["lo"], h["hi"]]
            repro["bootstrap_scene"][k][b]["max_abs_diff"] = float(np.max(np.abs(np.subtract(
                repro["bootstrap_scene"][k][b]["here"], repro["bootstrap_scene"][k][b]["stored"]))))
    out = dict(split="val", frames=len(items), scenes=len(scenes),
               recordings={r: sum(splits.recording(s) == r for s in scenes) for r in recordings},
               resamples=args.resamples, models={k: [v[0], str(v[1]), str(v[2])] for k, v in MODELS.items()},
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
