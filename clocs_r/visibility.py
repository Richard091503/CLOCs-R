"""Per-candidate camera-visibility inputs (results/next_round/PLAN.md, Phase 3), cached beside the cells of a split.

For each frame: the projected 2D height of every 3D candidate and the number of LiDAR points inside it (both
normalized, see FEATURES), and the score of the frame's 40th best 2D candidate, which sets the K40 pairing.

Usage: python -m clocs_r.visibility --split fit,select,val --cells runs/next_round/phase1/cells_in \
           --cand3d runs/next_round/phase1/cand_in/3d --cand2d runs/next_round/phase1/cand_in/2d
"""
import argparse
import multiprocessing as mp
import pathlib
import pickle

import numpy as np
from scipy.spatial import cKDTree

from clocs_r import aimotive, cache_cells, fusion, splits

FEATURES = ("height", "points")
POINTS_NORM = 8.0  # log(1 + n) / 8: about 1 at 3,000 points
TOP_2D = 40
MOVING = 1.0  # m/s: a radar return is moving if its ego-compensated radial speed is at least this (Phase 4 context)


def projected_height(seq, boxes):
    """Unclipped image height in pixels of each 3D box, and whether the camera can see it (aimotive.project_boxes)."""
    if len(boxes) == 0:
        return np.zeros(0, np.float32), np.zeros(0, bool)
    uv, _ = seq.project(aimotive.box_corners_3d(boxes.astype(np.float64)))
    _, valid = aimotive.project_boxes(seq, boxes)
    return (uv[..., 1].max(1) - uv[..., 1].min(1)).astype(np.float32), valid


def points_in_boxes(boxes, pts):
    """Number of points (N, >=3) inside each rotated 3D box (K, 7)."""
    count = np.zeros(len(boxes), np.int64)
    if len(boxes) == 0 or len(pts) == 0:
        return count
    radius = np.hypot(boxes[:, 3], boxes[:, 4]) / 2
    near = cKDTree(pts[:, :2]).query_ball_point(boxes[:, :2], radius)
    cand = np.repeat(np.arange(len(boxes)), [len(n) for n in near])
    if len(cand) == 0:
        return count
    p = pts[np.concatenate(near).astype(np.int64), :3]
    b = boxes[cand]
    dx, dy = p[:, 0] - b[:, 0], p[:, 1] - b[:, 1]
    c, s = np.cos(b[:, 6]), np.sin(b[:, 6])
    u, v = dx * c + dy * s, -dx * s + dy * c
    inside = (np.abs(u) <= b[:, 3] / 2) & (np.abs(v) <= b[:, 4] / 2) & (np.abs(p[:, 2] - b[:, 2]) <= b[:, 5] / 2)
    np.add.at(count, cand[inside], 1)
    return count


def features(seq, boxes, pts):
    """(K, 2) float32: normalized log height (MISSING if the camera cannot see the box) and normalized log point count."""
    h, valid = projected_height(seq, boxes)
    height = np.where(valid, np.log1p(h) / np.log1p(seq.meta["image_size"][1]), fusion.MISSING)
    return np.stack([height, np.log1p(points_in_boxes(boxes, pts)) / POINTS_NORM], 1).astype(np.float32)


def top_score(scores2d, k=TOP_2D):
    """The k-th highest 2D score of a frame, or MISSING if it has k or fewer 2D candidates (then every one is kept)."""
    return float(np.sort(scores2d)[-k]) if len(scores2d) > k else fusion.MISSING


def moving_returns(boxes, pts, vel, delta0=0.5, kappa=0.02, v_min=MOVING):
    """log(1 + moving radar returns) in each candidate's gate (the gate of radar_features.candidate_features): (K,)."""
    out = np.zeros(len(boxes), np.float32)
    moving = np.abs(vel) >= v_min
    if len(boxes) == 0 or not moving.any():
        return out
    p = pts[moving]
    delta = delta0 + kappa * np.hypot(boxes[:, 0], boxes[:, 1])
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    dx, dy = p[None, :, 0] - boxes[:, None, 0], p[None, :, 1] - boxes[:, None, 1]
    gate = (np.abs(dx * c[:, None] + dy * s[:, None]) <= boxes[:, None, 3] / 2 + delta[:, None]) & \
           (np.abs(-dx * s[:, None] + dy * c[:, None]) <= boxes[:, None, 4] / 2 + delta[:, None])
    return np.log1p(gate.sum(1)).astype(np.float32)


def _moving_work(job):
    path, frame, cand3d, delta0, kappa, sweeps = job
    seq = cache_cells._seq(path)
    with np.load(pathlib.Path(cand3d) / seq.name / f"{frame:07d}.npz") as z:
        boxes = z["boxes"]
    pts, vel = seq.radar_accumulated(frame, sweeps)
    return moving_returns(boxes, pts, vel, delta0, kappa).astype(np.float16)


def build_moving(split, cells, cand3d, delta0, kappa, sweeps=cache_cells.SWEEPS, workers=14):
    """log(1 + moving returns) per candidate, cached beside the radar features of the same gate."""
    path = pathlib.Path(cells) / f"{split}_moving_{cache_cells.radar_name(delta0, kappa, sweeps)}.pkl"
    if not path.exists():
        jobs = [(s.path, f, cand3d, delta0, kappa, sweeps) for s, f in splits.frames(split)]
        with mp.Pool(workers) as pool:
            out = pool.map(_moving_work, jobs, chunksize=16)
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=4)
    with open(path, "rb") as f:
        return pickle.load(f)


def _work(job):
    path, frame, cand3d, cand2d = job
    seq = cache_cells._seq(path)
    boxes, _, det2d = cache_cells.load_candidates(seq, frame, cand3d, cand2d)
    return features(seq, boxes, seq.lidar(frame)).astype(np.float16), top_score(det2d[:, 4]), (seq.name, frame)


def build(split, cells, cand3d, cand2d, workers=14):
    path = pathlib.Path(cells) / f"{split}_vis.pkl"
    if path.exists():
        return path
    jobs = [(s.path, f, cand3d, cand2d) for s, f in splits.frames(split)]
    with mp.Pool(workers) as pool:
        out = pool.map(_work, jobs, chunksize=16)
    with open(path, "wb") as f:
        pickle.dump(dict(features=[o[0] for o in out], top2d=np.array([o[1] for o in out], np.float32),
                         keys=[o[2] for o in out]), f, protocol=4)
    return path


def load(split, cells, base=None):
    """The cached visibility inputs of a split; checks that the frames line up with the base cells if given."""
    with open(pathlib.Path(cells) / f"{split}_vis.pkl", "rb") as f:
        vis = pickle.load(f)
    if base is not None:
        assert [(it["seq"], it["frame"]) for it in base] == vis["keys"], f"{split}: visibility and cells frames differ"
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="fit,select,val")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--cand3d", required=True)
    ap.add_argument("--cand2d", required=True)
    ap.add_argument("--moving", default="", help="delta0,kappa: also cache the moving radar returns for this gate")
    args = ap.parse_args()
    for split in args.split.split(","):
        print("cached", build(split, args.cells, args.cand3d, args.cand2d), flush=True)
        if args.moving:
            build_moving(split, args.cells, args.cand3d, *map(float, args.moving.split(",")))
            print("cached moving returns", split, flush=True)


if __name__ == "__main__":
    main()
