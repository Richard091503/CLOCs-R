"""Turn cached detector candidates into per-frame fusion inputs, in parallel.

For each frame: the pair cells without radar channels (Eq. 3), each candidate's five radar features for a given gate,
training targets, and the labels needed for evaluation. Radar channels are appended from the candidate features at
load time, so one cache serves every feature set in Table II, and only the radar part is rebuilt for a new gate.

Usage: python -m clocs_r.cache_cells --split train --delta0 0.5 --kappa 0.02
"""
import argparse
import multiprocessing as mp
import os
import pathlib
import pickle

import numpy as np

from clocs_r import aimotive, fusion, metrics, radar_features, splits

MAX_RANGE = 204.8  # normalizes d_j
POS_IOU, NEG_IOU = 0.7, 0.5  # BEV IoU thresholds for training targets (Section II-D)
SWEEPS = 3  # M: radar sweeps accumulated per frame


def load_candidates(seq, frame, cand3d, cand2d):
    with np.load(pathlib.Path(cand3d) / seq.name / f"{frame:07d}.npz") as z:
        boxes, scores = z["boxes"], z["scores"]
    det2d = np.load(pathlib.Path(cand2d) / seq.name / f"{frame:07d}.npy")
    return boxes, scores, det2d


def eval_labels(seq, frame, boxes):
    """Scored ground truth of a frame, the candidates that can be scored, and the candidates sitting on large vehicles."""
    gt, _ = seq.boxes(frame, splits.CAR_CLASSES)
    ignore, _ = seq.boxes(frame, splits.IGNORE_CLASSES)
    if len(gt):
        gt = gt[aimotive.in_region(gt) & aimotive.in_camera_view(seq, gt)]
    keep_eval = aimotive.in_region(boxes) & aimotive.in_camera_view(seq, boxes)
    on_ignored = np.zeros(len(boxes), bool)
    if len(ignore):  # candidates on trucks and buses are neither positives nor negatives, in training or evaluation
        on_ignored = metrics.iou_bev(ignore, boxes).max(0) >= 0.1
        keep_eval &= ~on_ignored
    return gt, keep_eval, on_ignored


def base_item(seq, frame, cand3d, cand2d):
    boxes, scores, det2d = load_candidates(seq, frame, cand3d, cand2d)
    proj, valid = aimotive.project_boxes(seq, boxes)
    proj = proj.astype(np.float32)
    proj[~valid] = np.nan
    dist = np.hypot(boxes[:, 0], boxes[:, 1]) / MAX_RANGE
    empty = np.zeros((len(scores), 0), np.float32)
    cells, column = fusion.build_cells(det2d[:, :4], det2d[:, 4], proj, scores, dist, empty)

    gt, keep_eval, on_ignored = eval_labels(seq, frame, boxes)
    target = np.zeros(len(scores), np.float32)
    weight = np.ones(len(scores), np.float32)
    best = metrics.iou_bev(gt, boxes).max(0) if len(gt) else np.zeros(len(scores))
    target[best >= POS_IOU] = 1.0
    weight[(best > NEG_IOU) & (best < POS_IOU)] = 0.0
    weight[on_ignored] = 0.0
    # The 3D score and distance of each cell are rebuilt from its candidate at load time, which saves more than half the disk
    return dict(pairs=cells[:, :2].astype(np.float32), column=column.astype(np.int16), dist=dist.astype(np.float32),
                boxes=boxes, scores=scores, target=target, weight=weight, gt=gt, keep_eval=keep_eval,
                seq=seq.name, odd=seq.odd, frame=frame)


def radar_item(seq, frame, cand3d, delta0, kappa, sweeps, motion=False):
    with np.load(pathlib.Path(cand3d) / seq.name / f"{frame:07d}.npz") as z:
        boxes = z["boxes"]
    pts, vel = seq.radar_accumulated(frame, sweeps, motion)
    return radar_features.candidate_features(boxes, pts, vel, delta0, kappa)


def _work(job):
    kind, seq_path, frame, a = job
    seq = _seq(seq_path)
    if kind == "base":
        return base_item(seq, frame, a["cand3d"], a["cand2d"])
    return radar_item(seq, frame, a["cand3d"], a["delta0"], a["kappa"], a["sweeps"], a["motion"])


_SEQS = {}


def _seq(path):
    if path not in _SEQS:
        _SEQS.clear()  # jobs arrive roughly in sequence order; keeps per-worker memory flat
        _SEQS[path] = aimotive.Sequence(path)
    return _SEQS[path]


def radar_name(delta0, kappa, sweeps, motion=False):
    return f"radar_d{delta0}_k{kappa}_m{sweeps}" + ("_motion" if motion else "")


def build(split, out, cand3d, cand2d, delta0, kappa, sweeps, limit=0, workers=14, use_radar=True, motion=False):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    items = [(seq.path, f) for seq, f in splits.frames(split)]
    if limit:
        items = items[:: max(1, len(items) // limit)][:limit]
    base_path = out / f"{split}_base.pkl"
    a = dict(cand3d=cand3d, cand2d=cand2d, delta0=delta0, kappa=kappa, sweeps=sweeps, motion=motion)
    with mp.Pool(workers) as pool:
        if not base_path.exists():
            base = pool.map(_work, [("base", s, f, a) for s, f in items], chunksize=16)
            with open(base_path, "wb") as fp:
                pickle.dump(base, fp, protocol=4)
        rad_path = out / f"{split}_{radar_name(delta0, kappa, sweeps, motion)}.pkl"
        if use_radar and not rad_path.exists():
            radar = pool.map(_work, [("radar", s, f, a) for s, f in items], chunksize=16)
            with open(rad_path, "wb") as fp:
                pickle.dump(np.stack(radar).astype(np.float16), fp, protocol=4)  # (frames, K, 5)


def load(split, out, delta0, kappa, sweeps, use_radar=True):
    out = pathlib.Path(out)
    with open(out / f"{split}_base.pkl", "rb") as fp:
        base = pickle.load(fp)
    if not use_radar:
        return base, [None] * len(base)
    return base, load_radar(split, out, delta0, kappa, sweeps)


def load_radar(split, out, delta0, kappa, sweeps, motion=False):
    with open(pathlib.Path(out) / f"{split}_{radar_name(delta0, kappa, sweeps, motion)}.pkl", "rb") as fp:
        return pickle.load(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val,holdout,train")
    ap.add_argument("--out", default="runs/cells")
    ap.add_argument("--cand3d", default="runs/candidates3d")
    ap.add_argument("--cand2d", default="runs/candidates2d")
    ap.add_argument("--delta0", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.02)
    ap.add_argument("--sweeps", type=int, default=SWEEPS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_radar", action="store_true", help="only the camera-LiDAR cells, enough for feature set R0")
    args = ap.parse_args()
    for split in args.split.split(","):
        build(split, args.out, args.cand3d, args.cand2d, args.delta0, args.kappa, args.sweeps, args.limit, use_radar=not args.no_radar)
        print("cached", split, flush=True)


if __name__ == "__main__":
    main()
