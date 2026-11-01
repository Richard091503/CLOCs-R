"""Why is SECOND's holdout AP what it is? Compares holdout with a train sample, by box error and by LiDAR points in the GT."""
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

from clocs_r import aimotive, cache_cells, metrics, splits
from clocs_r.second_dataset import points_in_box

POINT_LIMITS = (1, 5, 20)


def frame(job):
    seq_path, f, cand3d = job
    seq = cache_cells._seq(seq_path)
    path = pathlib.Path(cand3d) / seq.name / f"{f:07d}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        boxes, scores = z["boxes"], z["scores"]
    gt, mask, _ = cache_cells.eval_labels(seq, f, boxes)
    boxes, scores = boxes[mask], scores[mask]
    keep = metrics.nms_bev(boxes, scores, 0.1)
    pts = seq.lidar(f)
    npts = np.array([points_in_box(pts, b).sum() for b in gt], int)
    return dict(gt=gt, det=boxes[keep], score=scores[keep], npts=npts, odd=seq.odd)


def errors(records):
    """For each GT, the best-BEV-IoU detection above score 0.3: signed box errors, by distance bin."""
    rows = {}
    for r in records:
        if not len(r["gt"]) or not len(r["det"]):
            continue
        s = r["score"] >= 0.3
        if not s.any():
            continue
        det = r["det"][s]
        iou = metrics.iou_bev(r["gt"], det)
        best, idx = iou.max(1), iou.argmax(1)
        rg = np.hypot(r["gt"][:, 0], r["gt"][:, 1])
        for lo, hi in aimotive.DIST_BINS:
            m = (rg >= lo) & (rg < hi) & (best >= 0.3)
            if m.any():
                d, g = det[idx[m]], r["gt"][m]
                rows.setdefault(f"{lo}-{hi}", []).append(np.c_[d[:, :6] - g[:, :6], np.abs(np.arctan2(np.sin(d[:, 6] - g[:, 6]), np.cos(d[:, 6] - g[:, 6]))), best[m]])
    out = {}
    for k, v in rows.items():
        a = np.concatenate(v)
        out[k] = dict(n=len(a), **{n: [round(float(np.median(a[:, i])), 3), round(float(np.percentile(np.abs(a[:, i]), 75)), 3)]
                                   for i, n in enumerate(("dx", "dy", "dz", "dl", "dw", "dh", "|dyaw|", "bevIoU"))})
    return out


def with_points(records, limit):
    """Cars with fewer LiDAR points than limit become ignored, the way KITTI ignores boxes outside a difficulty."""
    out = []
    for r in records:
        ok = r["npts"] >= limit
        gt, ign = r["gt"][ok], r["gt"][~ok]
        det, score = r["det"], r["score"]
        if len(ign) and len(det):
            keep = metrics.iou_bev(ign, det).max(0) < 0.1
            det, score = det[keep], score[keep]
        out.append((gt, det, score))
    return out


def summarize(name, records):
    res = {}
    base = metrics.evaluate([(r["gt"], r["det"], r["score"]) for r in records], ioubev_thresholds=(0.3, 0.5, 0.7))
    res["all_gt"] = base
    for lim in POINT_LIMITS:
        res[f"npts>={lim}"] = metrics.evaluate(with_points(records, lim), ioubev_thresholds=(0.3, 0.5, 0.7))
    res["errors"] = errors(records)
    bins = ["all"] + [f"{lo}-{hi}" for lo, hi in aimotive.DIST_BINS]
    print("==", name, "frames", len(records))
    for key in ["all_gt"] + [f"npts>={l}" for l in POINT_LIMITS]:
        for b in bins:
            v = res[key][b]
            print(f"{key:9s} {b:8s} n_gt {v['n_gt']:6d} n_det {v['n_det']:6d} AP3D@.5 {v['AP3D@0.5']:.3f} AP3D@.7 {v['AP3D@0.7']:.3f} "
                  f"BEV@.3 {v['APBEV@0.3']:.3f} BEV@.5 {v['APBEV@0.5']:.3f} BEV@.7 {v['APBEV@0.7']:.3f} CD {v['APCD']:.3f}")
    print("errors [median, 75th pct of abs]:")
    for k, v in res["errors"].items():
        print(" ", k, v)
    npts = np.concatenate([r["npts"] for r in records])
    rg = np.concatenate([np.hypot(r["gt"][:, 0], r["gt"][:, 1]) for r in records])
    for lo, hi in aimotive.DIST_BINS:
        m = (rg >= lo) & (rg < hi)
        print(f"  GT {lo}-{hi}: {m.sum()} cars, 0 pts {np.mean(npts[m] == 0):.2f}, <5 pts {np.mean(npts[m] < 5):.2f}, median pts {np.median(npts[m]):.0f}")
    return res


def main():
    scratch = pathlib.Path(sys.argv[1])
    results = {}
    for name, split, cand in (("holdout", "holdout", "runs/candidates3d"), ("train_sample", "train", str(scratch / "cand3d_train"))):
        items = [(seq.path, f, cand) for seq, f in splits.frames(split)]
        with mp.Pool(14) as pool:
            records = [r for r in pool.map(frame, items, chunksize=16) if r is not None]
        results[name] = summarize(name, records)
    json.dump(results, open(scratch / "diag_second.json", "w"), indent=1)


if __name__ == "__main__":
    main()
