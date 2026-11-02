"""Post hoc tuning of the fusion stage, following results/clocs_radar/tuning/PLAN.md. Reads holdout only unless --splits says otherwise.

A configuration joins changes with "+": R2, T1 (far targets), T2 (residual), T3 (logit input), T4a or T4b (5 or 10 sweeps with motion
correction) and T5 (RCS). A "R0+" prefix replaces the radar channels with none.

Usage: python -m clocs_r.tune_fusion --configs R2,T1,T2 --out results/clocs_radar/tuning/selection.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import pickle

import numpy as np
import torch

from clocs_r import cache_cells, metrics, radar_features, splits, train_fusion

GATE = (0.5, 0.02)
BASE_SET = "R2"
FAR_FROM = 50.0  # m
FAR_POS, FAR_NEG = 0.5, 0.3  # BEV IoU thresholds beyond FAR_FROM for T1
RCS_INDEX = len(radar_features.FEATURES)  # the RCS channel follows the five features
SEEDS = (1, 2, 3)
DATA = {}


def parse(config):
    """Turn a name such as T1+T2 into its settings."""
    parts = config.split("+")
    assert set(parts) <= {"R0", "R2", "T1", "T2", "T3", "T4a", "T4b", "T5"}, config
    sweeps = 5 if "T4a" in parts else 10 if "T4b" in parts else cache_cells.SWEEPS
    return dict(radar="R0" not in parts, far_targets="T1" in parts, sweeps=sweeps, motion="T4a" in parts or "T4b" in parts, rcs="T5" in parts,
                net=dict(residual="T2" in parts, logit_input="T3" in parts))


def names_idx(cfg):
    if not cfg["radar"]:
        return []
    idx = [radar_features.FEATURES.index(f) for f in radar_features.FEATURE_SETS[BASE_SET]]
    return idx + [RCS_INDEX] if cfg["rcs"] else idx


def _far_target(job):
    path, frame, cand3d, cand2d = job
    seq = cache_cells._seq(path)
    boxes, scores, _ = cache_cells.load_candidates(seq, frame, cand3d, cand2d)
    gt, _, on_ignored = cache_cells.eval_labels(seq, frame, boxes)
    best = metrics.iou_bev(gt, boxes).max(0) if len(gt) else np.zeros(len(scores))
    far = np.hypot(boxes[:, 0], boxes[:, 1]) >= FAR_FROM
    pos, neg = np.where(far, FAR_POS, cache_cells.POS_IOU), np.where(far, FAR_NEG, cache_cells.NEG_IOU)
    target = (best >= pos).astype(np.float32)
    weight = np.ones(len(scores), np.float32)
    weight[(best > neg) & (best < pos)] = 0.0
    weight[on_ignored] = 0.0
    return target, weight


def far_targets(args):
    path = pathlib.Path(args.cells) / "train_far_targets.pkl"
    if not path.exists():
        jobs = [(s.path, f, args.cand3d, args.cand2d) for s, f in splits.frames("train")]
        with mp.Pool(14) as pool:
            result = pool.map(_far_target, jobs, chunksize=16)
        with open(path, "wb") as f:
            pickle.dump(result, f, protocol=4)
    with open(path, "rb") as f:
        return pickle.load(f)


def _rcs_item(job):
    path, frame, cand3d, sweeps, motion = job
    seq = cache_cells._seq(path)
    with np.load(pathlib.Path(cand3d) / seq.name / f"{frame:07d}.npz") as z:
        boxes = z["boxes"]
    pts, _, rcs = seq.radar_accumulated_rcs(frame, sweeps, motion)
    return radar_features.candidate_rcs(boxes, pts, rcs, *GATE)


def radar_data(split, cfg, args):
    """The radar array of a split for this configuration, built once and kept on disk."""
    if not cfg["radar"]:
        return None
    cache_cells.build(split, args.cells, args.cand3d, args.cand2d, *GATE, cfg["sweeps"], motion=cfg["motion"])
    radar = cache_cells.load_radar(split, args.cells, *GATE, cfg["sweeps"], cfg["motion"])
    if not cfg["rcs"]:
        return radar
    path = pathlib.Path(args.cells) / f"{split}_rcs_{cache_cells.radar_name(*GATE, cfg['sweeps'], cfg['motion'])}.pkl"
    if not path.exists():
        jobs = [(s.path, f, args.cand3d, cfg["sweeps"], cfg["motion"]) for s, f in splits.frames(split)]
        with mp.Pool(14) as pool:
            rcs = pool.map(_rcs_item, jobs, chunksize=16)
        with open(path, "wb") as f:
            pickle.dump(np.stack(rcs).astype(np.float16), f, protocol=4)
    with open(path, "rb") as f:
        return np.concatenate([radar, pickle.load(f)[..., None]], -1)


def run_seed(job):
    """In a forked child, so that CUDA is first used here and the parent's cached frames are shared."""
    name, cfg, seed, epochs, eval_splits, out = job
    torch.set_num_threads(1)
    device = torch.device("cuda")
    idx = names_idx(cfg)
    train = DATA["train_far"] if cfg["far_targets"] else DATA["train"]
    radar = DATA["radar_train"] if idx else [None] * len(train)
    net, log = train_fusion.fit(train, radar, idx, epochs, seed, device, f"{name} s{seed}", net_kwargs=cfg["net"])
    torch.save(dict(model=net.state_dict(), config=name, settings=cfg, gate=GATE), pathlib.Path(out) / f"{name}_s{seed}.pth")
    result = dict(seed=seed, final_loss=log[-1]["loss"])
    for split in eval_splits:
        result[split] = train_fusion.evaluate(net, DATA[split], DATA["radar_" + split] if idx else [None] * len(DATA[split]), idx, device)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="holdout")
    ap.add_argument("--weights", default="runs/tuning")
    ap.add_argument("--cells", default="runs/cells")
    ap.add_argument("--cand3d", default="runs/candidates3d")
    ap.add_argument("--cand2d", default="runs/candidates2d")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    eval_splits = args.splits.split(",")
    pathlib.Path(args.weights).mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(args.out)
    results = json.load(open(out)) if out.exists() else {}
    ctx = mp.get_context("fork")

    DATA["train"], _ = cache_cells.load("train", args.cells, *GATE, cache_cells.SWEEPS, use_radar=False)
    for split in eval_splits:
        DATA[split], _ = cache_cells.load(split, args.cells, *GATE, cache_cells.SWEEPS, use_radar=False)
    for name in args.configs.split(","):
        cfg = parse(name)
        if cfg["far_targets"] and "train_far" not in DATA:
            DATA["train_far"] = [dict(item, target=t, weight=w) for item, (t, w) in zip(DATA["train"], far_targets(args))]
        DATA["radar_train"] = radar_data("train", cfg, args)
        for split in eval_splits:
            DATA["radar_" + split] = radar_data(split, cfg, args)
        with ctx.Pool(len(SEEDS)) as pool:
            runs = pool.map(run_seed, [(name, cfg, s, args.epochs, eval_splits, args.weights) for s in SEEDS])
        results[name] = dict(settings=cfg, runs=runs)
        far = {split: [r[split]["overall"]["50-200"]["APCD"] for r in runs] for split in eval_splits}
        print(name, {k: f"{np.mean(v):.4f} ({[round(x, 4) for x in v]})" for k, v in far.items()}, flush=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
