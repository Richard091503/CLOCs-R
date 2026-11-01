"""Feasibility of radar as the proposal source at range, with LiDAR and camera as support. Holdout only, no training of detectors.

Radar returns are clustered into fixed-size boxes aligned with the ego heading. A logistic model, fit on other scenes, then scores each
box from its radar evidence and its LiDAR and camera support. Also scores aiMotive val over KITTI's range, for the dataset comparison.

Usage, from the repo root: python results/clocs_radar/diag_radar_first.py
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from diag_radar_limits import auc, fit_logistic, predict_logistic

from clocs_r import aimotive, cache_cells, fusion, metrics, radar_features, splits, train_fusion

OUT = pathlib.Path(__file__).parent / "diag_radar_first.json"
GATE = (0.5, 0.02, 3)
BOX = (4.0, 2.0, 1.5)  # median training car, as radar gives no extent
MOVING = 1.0  # m/s
REAL_DIST = 2.0  # m, center distance that counts as a real car when fitting the scores
FAR = 50.0
BINS = [(0, 30), (30, 50), (50, 100), (100, 200), (50, 200)]
device = torch.device("cuda")


def clusters(pts, vel, link=3.0, dv=3.0):
    """Connected components of returns close in position and in Doppler."""
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        d = np.hypot(pts[i + 1:, 0] - pts[i, 0], pts[i + 1:, 1] - pts[i, 1])
        for j in np.nonzero((d <= link) & (np.abs(vel[i + 1:] - vel[i]) <= dv))[0] + i + 1:
            parent[find(i)] = find(int(j))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def proposals(seq, frame):
    pts, vel = seq.radar_accumulated(frame, GATE[2])
    rows, feats = [], []
    if len(pts):
        for g in clusters(pts, vel):
            x, y = np.median(pts[g, 0]), np.median(pts[g, 1])
            rows.append([x, y, BOX[2] / 2, *BOX, 0.0])
            feats.append([np.log1p(len(g)), np.median(np.abs(vel[g])) / radar_features.V_MAX])
    boxes = np.array(rows, np.float32).reshape(-1, 7)
    feats = np.array(feats, np.float32).reshape(-1, 2)
    keep = aimotive.in_region(boxes) & aimotive.in_camera_view(seq, boxes) if len(boxes) else np.zeros(0, bool)
    return boxes[keep], feats[keep]


def center_dist(a, b):
    return np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])


def frame_rows(seq, item):
    """One frame: radar boxes with their evidence and support, plus the recall of each source on the labeled cars."""
    frame = item["frame"]
    boxes, feats = proposals(seq, frame)
    gt = item["gt"]
    ignore, _ = seq.boxes(frame, splits.IGNORE_CLASSES)
    if len(ignore) and len(boxes):
        off = metrics.iou_bev(ignore, boxes).max(0) < 0.1
        boxes, feats = boxes[off], feats[off]
    n = len(boxes)
    mask = np.nonzero(item["keep_eval"])[0]
    kept = mask[metrics.nms_bev(item["boxes"][mask], item["scores"][mask], 0.1)]
    second_boxes, second_scores = item["boxes"][kept], item["scores"][kept]

    lidar = seq.lidar(frame)
    lidar = lidar[(lidar[:, 2] > 0.3) & (lidar[:, 2] < 2.5)]
    s2d, n_lidar, s3d = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    if n:
        _, _, det2d = cache_cells.load_candidates(seq, frame, "runs/candidates3d", "runs/candidates2d")
        proj, _ = aimotive.project_boxes(seq, boxes)
        iou = fusion.iou_2d_matrix(det2d[:, :4], np.nan_to_num(proj.astype(np.float32)))
        if len(det2d):
            s2d = np.max((iou >= 0.3) * det2d[:, 4][:, None], 0)
        for j, b in enumerate(boxes):
            n_lidar[j] = ((np.abs(lidar[:, 0] - b[0]) <= 4) & (np.abs(lidar[:, 1] - b[1]) <= 2.5)).sum()
        near = center_dist(boxes, item["boxes"]) <= 4
        s3d = np.max(near * item["scores"][None, :], 1)
    real = center_dist(boxes, gt).min(1) <= REAL_DIST if len(gt) and n else np.zeros(n, bool)
    rho = np.hypot(boxes[:, 0], boxes[:, 1])
    return dict(boxes=boxes, table=np.c_[rho, feats, s2d, np.log1p(n_lidar), s3d, real], second_boxes=second_boxes, second_scores=second_scores, gt=gt, seq=item["seq"], odd=item["odd"],
                gt_cover=coverage(gt, boxes, second_boxes, second_scores, item["boxes"], lidar))


def coverage(gt, radar_boxes, second_boxes, second_scores, candidates, lidar):
    if not len(gt):
        return np.zeros((0, 6))
    def near(b):
        return center_dist(gt, b).min(1) <= 4 if len(b) else np.zeros(len(gt), bool)
    confident = second_boxes[second_scores >= 0.05]
    points = np.array([((np.abs(lidar[:, 0] - g[0]) <= g[3] / 2 + 0.5) & (np.abs(lidar[:, 1] - g[1]) <= g[4] / 2 + 0.5)).sum() for g in gt])
    return np.c_[np.hypot(gt[:, 0], gt[:, 1]), near(radar_boxes), near(second_boxes), near(confident), near(candidates), points]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(t, supports):
    """Radar evidence, plus the chosen supports, each also allowed to vary with distance."""
    rho = t[:, 0] / 200
    x = [t[:, 1], t[:, 2], (t[:, 2] > MOVING / radar_features.V_MAX).astype(float), rho]
    if "camera" in supports:
        x += [t[:, 3], t[:, 3] * rho]
    if "lidar" in supports:
        x += [t[:, 4], t[:, 4] * rho]
    if "second" in supports:
        x += [logit(t[:, 5]), logit(t[:, 5]) * rho]
    return np.stack(x, 1)


def cross_fit(x, y, scene, folds=5):
    out = np.zeros(len(y))
    for k in range(folds):
        test = scene % folds == k
        out[test] = predict_logistic(fit_logistic(x[~test], y[~test].astype(float)), x[test])
    return 1 / (1 + np.exp(-np.clip(out, -30, 30)))


def score_bins(records):
    r = metrics.evaluate(records, bins=BINS)
    return {k: dict(n_gt=v["n_gt"], APCD=v["APCD"], APCD_at_4m=v["APCD@4.0"], APBEV=v["APBEV@0.3"], recall=v["recall@BEV0.3"]) for k, v in r.items()}


def center_nms(boxes, scores, dist=3.0):
    order = np.argsort(-scores)
    alive = np.ones(len(order), bool)
    keep = []
    d = center_dist(boxes[order], boxes[order])
    for i in range(len(order)):
        if alive[i]:
            keep.append(order[i])
            alive &= d[i] > dist
    return np.array(keep, int)


def main():
    results = {}
    # 1. aiMotive val over KITTI's range
    base, _ = cache_cells.load("val", "runs/cells", *GATE, use_radar=False)
    kitti_bins = [(0, 70.4), (50, 70.4)]
    records = [(it["gt"], b, s) for it, b, s in train_fusion.predict(None, base, [None] * len(base), [], device)]
    keep = ("n_gt", "APBEV@0.3", "APCD", "AP3D@0.5", "AP3D@0.7")
    results["val_second_kitti_range"] = {k: {m: v[m] for m in keep} for k, v in metrics.evaluate(records, bins=kitti_bins).items()}
    r0 = []
    for s in (1, 2, 3):
        net = fusion.FusionNet(0).to(device)
        net.load_state_dict(torch.load(f"runs/fusion/R0_s{s}/fusion.pth", map_location=device)["model"])
        rec = [(it["gt"], b, x) for it, b, x in train_fusion.predict(net, base, [None] * len(base), [], device)]
        r0.append({k: {m: v[m] for m in keep} for k, v in metrics.evaluate(rec, bins=kitti_bins).items()})
    results["val_r0_kitti_range"] = r0
    print("val KITTI range", json.dumps(results["val_second_kitti_range"]), flush=True)
    print("R0", {k: round(float(np.mean([x[k]["APBEV@0.3"] for x in r0])), 3) for k in r0[0]}, flush=True)
    del base, records, rec

    # 2. Radar first, on holdout
    base, _ = cache_cells.load("holdout", "runs/cells", *GATE, use_radar=False)
    seqs = {s.name: s for s in map(aimotive.Sequence, splits.scenes("holdout"))}
    frames = [frame_rows(seqs[it["seq"]], it) for it in base]
    names = {n: i for i, n in enumerate(sorted({f["seq"] for f in frames}))}

    cover = np.concatenate([f["gt_cover"] for f in frames])
    results["coverage"] = {}
    for lo, hi in BINS[:4]:
        c = cover[(cover[:, 0] >= lo) & (cover[:, 0] < hi)]
        radar, second, confident, cand, pts = c[:, 1] > 0, c[:, 2] > 0, c[:, 3] > 0, c[:, 4] > 0, c[:, 5]
        results["coverage"][f"{lo}-{hi}"] = dict(cars=len(c), radar=float(radar.mean()), second_kept=float(second.mean()), second_confident=float(confident.mean()), any_candidate=float(cand.mean()),
                                                 radar_when_second_missed=float(radar[~confident].mean()) if (~confident).any() else None, missed_by_both=float((~radar & ~confident).mean()), under_5_lidar_points=float((pts < 5).mean()),
                                                 radar_when_under_5_points=float(radar[pts < 5].mean()) if (pts < 5).any() else None)
    print("coverage", json.dumps(results["coverage"]), flush=True)

    table = np.concatenate([f["table"] for f in frames])
    scene = np.concatenate([np.full(len(f["table"]), names[f["seq"]]) for f in frames])
    y = table[:, 6] > 0
    results["proposals"] = {}
    for lo, hi in BINS[:4]:
        m = (table[:, 0] >= lo) & (table[:, 0] < hi)
        t, yy = table[m], y[m]
        moving = t[:, 2] > MOVING / radar_features.V_MAX
        results["proposals"][f"{lo}-{hi}"] = dict(boxes=int(m.sum()), real=int(yy.sum()), share_real=float(yy.mean()), share_real_moving=float(yy[moving].mean()) if moving.any() else None, moving=float(moving.mean()),
                                                  auc_support=auc(t[:, 1], yy), auc_doppler=auc(t[:, 2], yy), auc_camera=auc(t[:, 3], yy), auc_lidar=auc(t[:, 4], yy), auc_second=auc(t[:, 5], yy))
    print("proposals", json.dumps(results["proposals"]), flush=True)

    methods = {"radar only": (), "radar + camera": ("camera",), "radar + LiDAR points": ("lidar",), "radar + camera + LiDAR points": ("camera", "lidar"), "radar + camera + LiDAR + SECOND": ("camera", "lidar", "second")}
    results["detection"] = {}
    second_records = [(f["gt"], f["second_boxes"], f["second_scores"]) for f in frames]
    results["detection"]["SECOND"] = score_bins(second_records)
    offsets = np.cumsum([0] + [len(f["table"]) for f in frames])
    fused = {}
    for name, supports in methods.items():
        p = np.zeros(len(table))
        for mask in ((table[:, 0] < FAR), (table[:, 0] >= FAR)):  # near and far ranges get their own weights
            p[mask] = cross_fit(design(table[mask], supports), y[mask], scene[mask])
        fused[name] = p
        results["detection"][name] = score_bins([(f["gt"], f["boxes"], p[offsets[i]:offsets[i + 1]]) for i, f in enumerate(frames)])
        print(name, {k: round(v["APCD"], 3) for k, v in results["detection"][name].items()}, flush=True)

    # SECOND with radar boxes added beyond 50 m. Scores are calibrated on other scenes so the two sources can be ranked together.
    second_table = np.concatenate([np.c_[np.hypot(f["second_boxes"][:, 0], f["second_boxes"][:, 1]), f["second_scores"], (center_dist(f["second_boxes"], f["gt"]).min(1) <= REAL_DIST) if len(f["gt"]) else np.zeros(len(f["second_scores"]), bool)] for f in frames])
    second_scene = np.concatenate([np.full(len(f["second_scores"]), names[f["seq"]]) for f in frames])
    z = logit(second_table[:, 1])
    x2 = np.c_[z, z ** 2, second_table[:, 0] / 200, z * second_table[:, 0] / 200]
    p2 = np.zeros(len(z))
    for mask in ((second_table[:, 0] < FAR), (second_table[:, 0] >= FAR)):
        p2[mask] = cross_fit(x2[mask], second_table[mask, 2] > 0, second_scene[mask])
    off2 = np.cumsum([0] + [len(f["second_scores"]) for f in frames])
    results["detection"]["SECOND, calibrated"] = score_bins([(f["gt"], f["second_boxes"], p2[off2[i]:off2[i + 1]]) for i, f in enumerate(frames)])
    for name in ("radar + camera + LiDAR points", "radar + camera + LiDAR + SECOND"):
        records = []
        for i, f in enumerate(frames):
            far = f["table"][:, 0] >= FAR
            boxes = np.concatenate([f["second_boxes"], f["boxes"][far]])
            scores = np.concatenate([p2[off2[i]:off2[i + 1]], fused[name][offsets[i]:offsets[i + 1]][far]])
            k = center_nms(boxes, scores) if len(boxes) else np.zeros(0, int)
            records.append((f["gt"], boxes[k], scores[k]))
        results["detection"]["SECOND plus " + name] = score_bins(records)
        print("SECOND plus " + name, {k: round(v["APCD"], 3) for k, v in results["detection"]["SECOND plus " + name].items()}, flush=True)
    results["params"] = dict(box=BOX, moving_mps=MOVING, real_center_dist=REAL_DIST, far=FAR, sweeps=GATE[2])
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    for name, r in results["detection"].items():
        print(name.ljust(46), " ".join(f"{k}: CD {v['APCD']:.3f} @4m {v['APCD_at_4m']:.3f} BEV {v['APBEV']:.3f}" for k, v in r.items() if k in ("50-100", "100-200", "50-200")), flush=True)


if __name__ == "__main__":
    main()
